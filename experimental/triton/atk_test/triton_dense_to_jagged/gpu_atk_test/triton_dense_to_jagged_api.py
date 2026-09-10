# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

"""
dense_to_jagged 功能：
    将规则的 dense tensor 转换为不规则的 jagged tensor。
    例如：dense [B, seq, D] -> jagged [total_elements, D]

支持的输入维度：
    - 3D dense [B, seq, D]: 生成 2D jagged tensor，jagged_offsets 长度 1
    - 4D dense [B1, B2, seq, D]: 生成 3D jagged tensor，jagged_offsets 长度 2

限制条件（根据 limit.txt）：
1. dense 是 3 维或 4 维，最后一维 D（特征维度）<= 8192
2. 3D dense [B, seq, D]: B < seq < D < 8192
   - jagged_offsets 长度 1，shape [B+1]
   - jagged_offsets[0][0] = 0（偏移从 0 开始）
   - jagged_offsets[0][-1] <= B*seq（最后一个偏移不超过总元素数）
   - 相邻差值不超过 seq（每个 batch 元素的长度不超过 seq）
3. 4D dense [B1, B2, seq, D]: B1 < B2 < seq < D < 8192
   - jagged_offsets 长度 2，shape [B1+1] 和 [B1*B2+1]
   - jagged_offsets[0][0] = 0, jagged_offsets[0][-1] = B1*B2，相邻差值不超过 B2
   - jagged_offsets[1][0] = 0, jagged_offsets[1][-1] <= B1*B2*seq，相邻差值不超过 seq
4. operation_jagged_values 的数据类型与 dense 一致
"""
import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def get_device():
    """获取可用的设备，优先 NPU，其次 CUDA，最后 CPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'


# 从被测代码直接引用 triton kernel
from triton_dense_to_jagged import triton_dense_to_jagged, _jagged_offsets_to_dense_indice


def _generate_jagged_offsets(dense: torch.Tensor, jagged_offsets_input: list, device: str) -> list:
    """根据 dense 的 shape 生成满足约束的 jagged_offsets

    这个函数用于在 API 初始化时，根据 dense 的实际 shape 生成符合约束的 offset。
    使用确定性方式生成 offset，确保不同机器上结果一致。

    参数:
        dense: 输入的 dense tensor
        jagged_offsets_input: 从 yaml 生成的原始 offset 列表
        device: 目标设备（'npu'/'cuda'/'cpu'）

    返回:
        调整后的 offset 列表
    """
    dense_shape = list(dense.shape)
    dense_dim = len(dense_shape)

    adjusted_offsets = []

    if dense_dim == 3:
        # ==================== 3D dense [B, seq, D] ====================
        # 对应 1 个 offset，shape [B+1]
        # 约束：2*B <= offset[-1] <= B*seq，相邻差值不超过 seq
        B = dense_shape[0]
        seq = dense_shape[1]

        if len(jagged_offsets_input) >= 1:
            offset0 = jagged_offsets_input[0].to(device)

            # 计算目标最后一个值：取中间值
            min_last = 2 * B
            max_last = B * seq
            target_last = (min_last + max_last) // 2

            # 每个元素的平均长度
            avg_len = max(target_last // B, 1)
            avg_len = min(avg_len, seq)  # 不超过 seq

            # 分配长度
            remainder = target_last - avg_len * B
            segment_lengths = []
            for i in range(B):
                if i < remainder:
                    seg_len = avg_len + 1
                else:
                    seg_len = avg_len
                segment_lengths.append(seg_len)

        cumulative = [0]
        for seg_len in segment_lengths:
            cumulative.append(cumulative[-1] + seg_len)

            offset0 = torch.tensor(cumulative, dtype=offset0.dtype, device=device)
            adjusted_offsets.append(offset0)

    elif dense_dim == 4:
        # ==================== 4D dense [B1, B2, seq, D] ====================
        # 对应 2 个 offset
        # offset0: shape [B1+1]
        # offset1: shape [B1*B2+1]
        B1 = dense_shape[0]
        B2 = dense_shape[1]
        seq = dense_shape[2]

        if len(jagged_offsets_input) >= 1:
            # 第一个 offset：对应 B1 维度
            # 约束：jagged_offsets[0][-1] = B1*B2，相邻差值不超过 B2
            offset0 = jagged_offsets_input[0].to(device)

            # 每个 batch 元素应该有 B2 个元素
            segment_lengths = [B2] * B1
            cumulative = [0]
            for seg_len in segment_lengths:
                cumulative.append(cumulative[-1] + seg_len)

            offset0 = torch.tensor(cumulative, dtype=offset0.dtype, device=device)
            adjusted_offsets.append(offset0)

        if len(jagged_offsets_input) >= 2:
            # 第二个 offset：对应 B1*B2 维度
            # 约束：2*B1*B2 <= jagged_offsets[1][-1] <= B1*B2*seq，相邻差值不超过 seq
            offset1 = jagged_offsets_input[1].to(device)

            total_inner = B1 * B2
            # 取中间值：2*B1*B2 和 B1*B2*seq 的中间
            min_last = 2 * total_inner
            max_last = total_inner * seq
            target_last = (min_last + max_last) // 2

            # 每个元素的平均长度
            avg_len = max(target_last // total_inner, 1)
            avg_len = min(avg_len, seq)  # 不超过 seq

            # 分配长度：前几个元素多分配1
            remainder = target_last - avg_len * total_inner
            segment_lengths = []
            for i in range(total_inner):
                if i < remainder:
                    seg_len = avg_len + 1
                else:
                    seg_len = avg_len
                segment_lengths.append(seg_len)

            cumulative = [0]
            for seg_len in segment_lengths:
                cumulative.append(cumulative[-1] + seg_len)

            offset1 = torch.tensor(cumulative, dtype=offset1.dtype, device=device)
            adjusted_offsets.append(offset1)
    else:
        # 其他维度，直接使用
        adjusted_offsets = [offset.to(device) for offset in jagged_offsets_input]

    # 确保第一个元素为 0（偏移从 0 开始）
    for offset in adjusted_offsets:
        if offset[0].item() != 0:
            offset[0] = torch.tensor(0, dtype=offset.dtype, device=device)

    return adjusted_offsets


@register("triton_dense_to_jagged")
class TritonDenseToJaggedApi(BaseApi):
    """dense_to_jagged 算子的 ATK API 封装

    继承 BaseApi，实现算子的初始化和执行。
    """

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作

        在调用算子前执行一次，用于：
        1. 获取输入参数
        2. 生成满足约束的 jagged_offsets
        3. 准备输出 tensor
        4. 计算 kernel 所需的参数
        """
        device = get_device()

        # 同步操作，确保之前的所有操作完成
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ==================== 获取输入参数 ====================
        # 从 InputDataset 中获取 kwargs 形式的参数
        dense_input = input_data.kwargs.get("dense", None)
        jagged_offsets_input = input_data.kwargs.get("jagged_offsets", [])
        operation_function_input = input_data.kwargs.get("operation_function", None)
        operation_jagged_value_input = input_data.kwargs.get("operation_jagged_value", None)

        if dense_input is None:
            raise ValueError("dense not found")

        # ==================== 转换 dense 到正确设备 ====================
        if not isinstance(dense_input, torch.Tensor):
            dense = torch.tensor(dense_input, dtype=torch.float32, device=device)
        else:
            dense = dense_input.to(device)

        # ==================== 处理 jagged_offsets ====================
        jagged_offsets = []
        for offset_tensor in jagged_offsets_input:
            if isinstance(offset_tensor, torch.Tensor):
                jagged_offsets.append(offset_tensor.to(device))
            else:
                jagged_offsets.append(torch.tensor(offset_tensor, device=device))

        # ==================== 生成满足约束的 jagged_offsets ====================
        jagged_offsets = _generate_jagged_offsets(dense, jagged_offsets, device)

        # ==================== 计算输出的 jagged tensor 的大小 ====================
        # 最后一个 offset 的最后一个值就是输出的行数
        actual_total_valid = jagged_offsets[-1][-1].item()

        # ==================== 处理 operation_function ====================
        # operation_function 可以是 null, "add", 或 "mul"
        operation_function = None
        op_func_str = str(operation_function_input) if operation_function_input is not None else ""
        if op_func_str.lower() not in ['null', 'none', '']:
            operation_function = op_func_str

        # ==================== 处理 operation_jagged_values ====================
        # 当 operation_function 不为 null 时，需要生成 operation_jagged_values
        # 在确认输出的 jagged_tensor 的 shape 之后，再生成相同 shape，
        # 值都为 operation_jagged_value 的 tensor
        operation_jagged_values = None
        if operation_function is not None and operation_function not in ['null', 'None', '']:
            if operation_jagged_value_input is None:
                # 关闭融合以避免 kernel 对 None 指针做算术
                operation_function = None
            else:
                op_value = float(operation_jagged_value_input)
                # 生成与 output_jagged_value 相同 shape 的 tensor，值都为 op_value
                # 数据类型与 dense 保持一致
                operation_jagged_values = torch.full(
                    (int(actual_total_valid), dense.size(-1)),
                    op_value,
                    dtype=dense.dtype,
                    device=device
                )

        # ==================== 创建输出 tensor ====================
        output_jagged_value = torch.empty(
            (int(actual_total_valid), dense.size(-1)),
            device=device,
            dtype=dense.dtype,
        )

        # ==================== 计算 kernel 参数 ====================
        # grid: triton kernel 的启动配置，等于 offset 的元素数减 1
        grid = (jagged_offsets[-1].size(0) - 1,)

        # JAGGED_DIM: jagged tensor 的维度，等于 offset 数量 + 1
        JAGGED_DIM = len(jagged_offsets) + 1

        dense_indices = None
        if len(jagged_offsets) > 1:
            dense_indices = _jagged_offsets_to_dense_indice(
                jagged_offsets,
                dense.stride()[:-2],
                dense.size()[:-2],
            )
            if device == 'npu':
                dense_indices = dense_indices.npu()
            elif device == 'cuda':
                dense_indices = dense_indices.cuda()

        # 获取 dense 的 stride 信息，用于 kernel 计算
        dense_col_stride = dense.stride(-1)      # 最后一维的 stride
        dense_row_stride = dense.stride(-2)      # 倒数第二维的 stride
        dense_matrix_stride = dense.stride(-3)   # 倒数第三维的 stride

        # thread block 配置（triton 并行配置）
        thread_block_row_size = 32
        thread_block_col_size = 32

        # ==================== 保存实例变量 ====================
        # 保存所有需要用到变量，供 __call__ 使用
        self._device = device
        self._output_jagged_value = output_jagged_value
        self._jagged_offsets = jagged_offsets
        self._grid = grid
        self._JAGGED_DIM = JAGGED_DIM
        self._dense_indices = dense_indices
        self._dense = dense
        self._dense_col_stride = dense_col_stride
        self._dense_row_stride = dense_row_stride
        self._dense_matrix_stride = dense_matrix_stride
        self._thread_block_row_size = thread_block_row_size
        self._thread_block_col_size = thread_block_col_size
        self._operation_function = operation_function
        self._operation_jagged_values = operation_jagged_values

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        执行 dense_to_jagged 算子

        每次测试用例执行时都会调用这个函数。
        """
        device = self._device

        # 同步操作
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 调用 triton kernel
        triton_dense_to_jagged[self._grid](
            self._output_jagged_value,           # 输出：jagged tensor
            self._jagged_offsets[-1],            # 输入：offsets
            self._output_jagged_value.stride(0), # 输出 tensor 的 stride
            self._dense,                         # 输入：dense tensor
            self._dense_indices,                 # dense indices
            self._dense_col_stride,              # dense 最后一维 stride
            self._dense_row_stride,              # dense 倒数第二维 stride
            self._dense_matrix_stride,           # dense 倒数第三维 stride
            self._JAGGED_DIM,                    # jagged 维度数
            self._thread_block_row_size,         # thread block 行大小
            self._thread_block_col_size,         # thread block 列大小
            operation_function=self._operation_function,
            operation_jagged_value_ptr=self._operation_jagged_values,
        )

        # 同步操作
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output_jagged_value
