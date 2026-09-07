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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 从被测代码直接引用函数
from triton_jagged_to_dense import triton_jagged_to_dense


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _jagged_offsets_to_dense_indice(
    offsets: list[torch.Tensor], dense_strides: list[int], dense_sizes: list[int]
) -> torch.Tensor:
    """计算jagged offsets对应的dense indices"""
    device = get_device()
    output_offset = torch.zeros(len(offsets[-1]) - 1, device="cpu", dtype=torch.int32)

    offsets_cpu = [offset.cpu() for offset in offsets]
    n_levels = len(offsets_cpu)

    for entry in range(len(offsets_cpu[-1]) - 1):
        cursor = entry
        total = 0
        within_dense = True

        for lv in range(n_levels - 2, -1, -1):
            off = offsets_cpu[lv]
            lb, ub = 0, off.size(0)
            while lb < ub:
                middle = (lb + ub) // 2
                if off[middle] > cursor:
                    ub = middle
                else:
                    lb = middle + 1

            span = cursor - off[lb - 1]
            if dense_sizes and span >= dense_sizes[lv + 1]:
                within_dense = False
                break
            total += span * dense_strides[lv + 1]
            cursor = lb - 1

        if not within_dense:
            output_offset[entry] = -1
            continue

        total += cursor * dense_strides[0]
        if dense_sizes and cursor > dense_sizes[0]:
            total = -1
        output_offset[entry] = total

    if device == 'npu':
        return output_offset.npu()
    elif device == 'cuda':
        return output_offset.cuda()
    return output_offset


def _generate_jagged_offsets_by_shapes(shapes: list, total_elements: int, device: str) -> list:
    """
    根据已知的 shapes 生成 jagged_offsets（确定性算法）

    Args:
        shapes: 每个 jagged_offsets 的 shape 列表，如 [[4], [8], [16]]
        total_elements: 总元素数
        device: 设备

    Returns:
        jagged_offsets 列表
    """
    offsets = []

    # 计算每层的分段数
    layer_sizes = [shape[0] - 1 for shape in shapes]  # 每个 offset 的长度-1就是分段数

    # 生成最内层 offsets
    inner_size = layer_sizes[-1] if layer_sizes else 1
    inner_offsets = [0]
    current = 0
    step = max(1, total_elements // inner_size)
    for i in range(inner_size):
        seg_len = step + (i % 2)
        seg_len = max(1, seg_len)
        if current + seg_len > total_elements:
            seg_len = total_elements - current
        current += seg_len
        inner_offsets.append(current)
        if current >= total_elements:
            break
    if inner_offsets[-1] != total_elements:
        inner_offsets[-1] = total_elements
    offsets.append(torch.tensor(inner_offsets, dtype=torch.int64, device=device))

    # 生成外层 offsets
    for level in range(1, len(layer_sizes)):
        num_segments = layer_sizes[-(level + 1)]
        level_offsets = [0]
        current = 0
        inner_segments = len(offsets[-1]) - 1
        step = max(1, inner_segments // num_segments) if num_segments > 0 else 1
        for i in range(num_segments):
            seg_len = step + (i % 2)
            seg_len = max(1, seg_len)
            if current + seg_len > inner_segments:
                seg_len = inner_segments - current
            current += seg_len
            level_offsets.append(current)
            if current >= inner_segments:
                break
        if level_offsets[-1] != num_segments:
            level_offsets[-1] = num_segments
        offsets.append(torch.tensor(level_offsets, dtype=torch.int64, device=device))

    offsets = list(reversed(offsets))

    return offsets


@register("triton_jagged_to_dense")
class TritonJaggedToDenseApi(BaseApi):
    """jagged_to_dense 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """初始化操作"""
        # pylint: disable=attribute-defined-outside-init
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 获取输入参数
        jagged_values_input = input_data.kwargs.get("jagged_values", None)
        jagged_offsets_input = input_data.kwargs.get("jagged_offsets", None)  # 从输入获取 jagged_offsets
        jagged_max_lengths_input = input_data.kwargs.get("jagged_max_lengths", [8, 8])
        padding_value = input_data.kwargs.get("padding_value", 0.0)
        operation_function_input = input_data.kwargs.get("operation_function", None)
        operation_value_input = input_data.kwargs.get("operation_value", None)

        if jagged_values_input is None:
            raise ValueError("jagged_values not found")

        # 处理 jagged_values
        if not isinstance(jagged_values_input, torch.Tensor):
            jagged_values = torch.tensor(jagged_values_input, dtype=torch.float32, device=device)
        else:
            jagged_values = jagged_values_input.to(device)

        # 处理 jagged_max_lengths
        if isinstance(jagged_max_lengths_input, list):
            max_lengths = jagged_max_lengths_input
        elif hasattr(jagged_max_lengths_input, 'range_values'):
            max_lengths = list(jagged_max_lengths_input.range_values)
        else:
            max_lengths = [8, 8]

        # ====== 处理 jagged_offsets（根据输入的 shape 填充值）======
        # jagged_offsets_input 是一个列表，里面是 tensor
        # 例如：[tensor([0, 3, 7, 10]), tensor([0, 1, 2])]
        # 需要获取每个 tensor 的 shape，然后用确定性算法生成符合约束的值
        if jagged_offsets_input is not None and isinstance(jagged_offsets_input, list):
            # 获取每个 tensor 的 shape
            jagged_offsets_shapes = []
            for offset_tensor in jagged_offsets_input:
                if isinstance(offset_tensor, torch.Tensor):
                    jagged_offsets_shapes.append(list(offset_tensor.shape))
                elif hasattr(offset_tensor, 'shape'):
                    jagged_offsets_shapes.append(list(offset_tensor.shape))

            if jagged_offsets_shapes:
                # 根据已知的 shape，使用确定性算法生成偏移值
                jagged_offsets = _generate_jagged_offsets_by_shapes(jagged_offsets_shapes, jagged_values.shape[0], device)
            else:
                raise ValueError("Cannot get shapes from jagged_offsets_input")
        else:
            # jagged_offsets 必须提供
            raise ValueError("jagged_offsets is required, cannot be None")

        # 确保 offsets 约束
        total_elements = jagged_values.shape[0]
        for offset in jagged_offsets:
            if offset[0].item() != 0:
                offset[0] = torch.tensor(0, device=device)

        if jagged_offsets[-1][-1].item() != total_elements:
            jagged_offsets[-1][-1] = torch.tensor(total_elements, device=device)

        for i in range(len(jagged_offsets) - 1):
            expected = jagged_offsets[i + 1].size(0) - 1
            if jagged_offsets[i][-1].item() != expected:
                jagged_offsets[i][-1] = torch.tensor(expected, device=device)

        # 根据 offsets 计算 jagged_max_lengths的值
        computed_max_lengths = []
        for offset in jagged_offsets:
            offset_cpu = offset.cpu()
            diffs = [offset_cpu[j + 1].item() - offset_cpu[j].item() for j in range(len(offset_cpu) - 1)]
            computed_max_lengths.append(max(diffs) if diffs else 1)
        max_lengths = computed_max_lengths

        # 根据 operation_function 和 operation_value 生成 operation_dense
        operation_dense = None
        operation_function = None
        op_func_str = str(operation_function_input) if operation_function_input is not None else ""
        if op_func_str.lower() not in ['null', 'none', ''] and operation_value_input is not None:
            B = jagged_offsets[0].size(0) - 1
            output_shape = [B] + max_lengths + [jagged_values.shape[-1]]
            op_value = float(operation_value_input)
            operation_dense = torch.full(output_shape, op_value, dtype=jagged_values.dtype, device=device)
            operation_function = op_func_str

        # 保存实例变量
        self._device = device
        self._jagged_values = jagged_values
        self._jagged_offsets = jagged_offsets
        self._jagged_max_lengths = max_lengths
        self._padding_value = padding_value
        self._operation_function = operation_function
        self._operation_dense = operation_dense

        # 创建 output_dense 并计算 dense_indices
        outer_dense_size = len(jagged_offsets[0]) - 1
        inner_dense_size = jagged_values.size(-1)
        JAGGED_DIM = len(jagged_offsets) + 1

        output_shape = (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,)
        self._output_dense = torch.full(output_shape, padding_value, dtype=jagged_values.dtype, device=device)

        self._dense_indices = _jagged_offsets_to_dense_indice(
            jagged_offsets,
            self._output_dense.stride()[:-2],
            self._output_dense.size()[:-2],
        )

        self._JAGGED_DIM = JAGGED_DIM
        self._thread_block_row_size = 32
        self._thread_block_col_size = 32
        self._dense_col_stride = self._output_dense.stride(-1)
        self._dense_row_stride = self._output_dense.stride(-2)
        self._dense_matrix_stride = self._output_dense.stride(-3)
        self._grid = (len(jagged_offsets[-1]) - 1,)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 jagged_to_dense 算子"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        triton_jagged_to_dense[self._grid](
            self._jagged_values,
            self._jagged_offsets[-1],
            self._jagged_values.stride(0),
            self._output_dense,
            self._dense_indices,
            self._dense_col_stride,
            self._dense_row_stride,
            self._dense_matrix_stride,
            self._JAGGED_DIM,
            self._thread_block_row_size,
            self._thread_block_col_size,
            operation_function=self._operation_function,
            operation_dense=self._operation_dense,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output_dense
