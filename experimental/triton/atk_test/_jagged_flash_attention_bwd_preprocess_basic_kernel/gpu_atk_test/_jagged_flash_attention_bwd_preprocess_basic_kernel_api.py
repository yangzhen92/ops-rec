# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
jagged_flash_attention_bwd_preprocess_basic_kernel ATK API - 双精度测试版本

强制 fp32 转换，用于双标杆精度测试。

语义：Flash Attention 反向传播预处理
    delta[i] = sum(o[i, :] * do[i, :])
"""
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 导入 Triton kernel
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triton_jagged_flash_attention_basic import _jagged_flash_attention_bwd_preprocess_basic_kernel


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'


def _generate_jagged_offset_by_shape(shape, total_elements, device):
    """
    确定性生成 jagged offset
    offset[0] = 0, offset[-1] = total_elements
    中间值均匀分布
    """
    expected_length = shape[0]
    B = expected_length - 1

    if B <= 0:
        return torch.tensor([0, total_elements], dtype=torch.int64, device=device)

    offsets = [0]
    step = total_elements // B
    current = 0

    for i in range(B):
        # 确定性变化：交替 +0 和 +1
        seg_len = step + (i % 2)
        seg_len = max(1, seg_len)

        if current + seg_len > total_elements:
            seg_len = total_elements - current

        current += seg_len
        offsets.append(current)

        if current >= total_elements:
            break

    # 确保生成正确数量的 offset
    while len(offsets) < expected_length:
        offsets.append(total_elements)

    # 确保最后一个值是 total_elements
    if offsets[-1] != total_elements:
        offsets[-1] = total_elements

    result = torch.tensor(offsets, dtype=torch.int64, device=device)

    # 验证
    assert result.shape[0] == expected_length, f"offset length mismatch: {result.shape[0]} != {expected_length}"
    assert result[0].item() == 0, f"offset[0] should be 0, got {result[0].item()}"
    assert result[-1].item() == total_elements, f"offset[-1] should be {total_elements}, got {result[-1].item()}"

    return result


@register("_jagged_flash_attention_bwd_preprocess_basic_kernel")
class JaggedFlashAttentionBwdPreprocessBasicKernelApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        print('look! init_by_input_data') #调试用，随后删
        """预处理阶段（不计入性能）"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 从 input_data 获取参数
        o_input = input_data.kwargs.get("o", None)
        do_input = input_data.kwargs.get("do", None)
        o_offset_input = input_data.kwargs.get("o_offset", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", None)
        BLOCK_SIZE_M = input_data.kwargs.get("BLOCK_SIZE_M", None)
        BLOCK_SIZE_D = input_data.kwargs.get("BLOCK_SIZE_D", None)

        if o_input is None or do_input is None:
            raise ValueError("o and do are required")

        # ====== NPU dtype 适配（Fix）======
        # 背景：kernel 内 tl.load(other=...) 等在 Ascend 上可能被促升为 fp64，
        #       导致编译期 dtype 报错。统一强转 fp32 让 kernel 编译通过；
        #       输出再转回原始 dtype，与 GPU golden 双标杆 dtype 一致。
        # 记录原始 dtype（以 o 为准）
        if isinstance(o_input, torch.Tensor):
            self._orig_dtype = o_input.dtype
            print('look! if isinstance(o_input, torch.Tensor)') #调试用，随后删
            print(o_input.dtype) #调试用，随后删
        else:
            self._orig_dtype = torch.float32
            print('look! else') #调试用，随后删
        print(self._orig_dtype) #调试用，随后删

        # 转换 tensor 到目标设备（强转 fp32）
        # 原始（未强转）实现：o = o_input.to(device) / do = do_input.to(device)
        if not isinstance(o_input, torch.Tensor):
            o = torch.tensor(o_input, dtype=torch.float32, device=device)
        else:
            o = o_input.to(device=device, dtype=torch.float32)

        if not isinstance(do_input, torch.Tensor):
            do = torch.tensor(do_input, dtype=torch.float32, device=device)
        else:
            do = do_input.to(device=device, dtype=torch.float32)

        # 确保连续性
        if not o.is_contiguous():
            o = o.contiguous()
        if not do.is_contiguous():
            do = do.contiguous()

        # 获取维度
        total_seq = o.shape[0]
        D = o.shape[1]

        # 验证 o 和 do shape 一致
        if o.shape != do.shape:
            raise ValueError(f"o.shape {o.shape} != do.shape {do.shape}")

        # 生成或处理 o_offset
        if o_offset_input is None:
            # 默认生成 offset
            B = 8
            o_offset = _generate_jagged_offset_by_shape([B + 1], total_seq, device)
        else:
            if isinstance(o_offset_input, list):
                # 获取 shape 并生成
                if len(o_offset_input) > 0 and hasattr(o_offset_input[0], 'shape'):
                    offset_shape = list(o_offset_input[0].shape)
                else:
                    offset_shape = [9]  # 默认 B=8
                o_offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
            elif isinstance(o_offset_input, torch.Tensor):
                o_offset = o_offset_input.to(device)
                if o_offset[-1].item() > total_seq:
                    # 重新生成
                    o_offset = _generate_jagged_offset_by_shape(list(o_offset.shape), total_seq, device)
            else:
                # 默认
                B = 8
                o_offset = _generate_jagged_offset_by_shape([B + 1], total_seq, device)

        B = o_offset.shape[0] - 1

        # 处理列表格式的参数（ATK 可能传入 [value] 而不是 value）
        if isinstance(max_seq_len, list):
            max_seq_len = max_seq_len[0] if len(max_seq_len) > 0 else None
        if isinstance(BLOCK_SIZE_M, list):
            BLOCK_SIZE_M = BLOCK_SIZE_M[0] if len(BLOCK_SIZE_M) > 0 else None
        if isinstance(BLOCK_SIZE_D, list):
            BLOCK_SIZE_D = BLOCK_SIZE_D[0] if len(BLOCK_SIZE_D) > 0 else None

        # 处理默认值：如果参数为 None，使用合理的默认值
        if max_seq_len is None or max_seq_len <= 0:
            # 根据实际数据推算合理的 max_seq_len
            # 保守估计：total_seq / B，向上取整到 2 的幂次方
            if B > 0:
                avg_seq = (total_seq + B - 1) // B
                # 向上取整到最近的 2 的幂次方
                max_seq_len = 1
                while max_seq_len < avg_seq:
                    max_seq_len *= 2
                max_seq_len = max(16, max_seq_len)  # 至少 16
            else:
                max_seq_len = 128  # 默认值

        if BLOCK_SIZE_M is None or BLOCK_SIZE_M <= 0:
            BLOCK_SIZE_M = 32

        if BLOCK_SIZE_D is None or BLOCK_SIZE_D <= 0:
            BLOCK_SIZE_D = 64

        # 关键约束：BLOCK_SIZE_D 必须 >= D
        # kernel 没有对 D 维度循环，一次性处理所有 D 个元素
        if BLOCK_SIZE_D < D:
            # 向上取整到最近的 2 的幂次方
            BLOCK_SIZE_D = 1
            while BLOCK_SIZE_D < D:
                BLOCK_SIZE_D *= 2

        # 关键修复：ATK 可能传入非法的 offset 数据（包含负数、非递增等）
        # 参考其他 jagged 算子的做法：只使用 offset 的 shape，重新生成合法数据
        if isinstance(o_offset, torch.Tensor) and o_offset.shape[0] > 1:
            offset_shape = list(o_offset.shape)
            o_offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
            B = o_offset.shape[0] - 1

        # 创建输出 tensor
        delta = torch.zeros(total_seq, dtype=torch.float32, device=device)

        # 计算 grid
        import triton
        grid = (
            triton.cdiv(max_seq_len, BLOCK_SIZE_M),
            B,
        )

        # 保存实例变量
        self._device = device
        self._o = o
        self._do = do
        self._o_offset = o_offset
        self._delta = delta
        self._max_seq_len = max_seq_len
        self._D = D
        self._BLOCK_SIZE_M = BLOCK_SIZE_M
        self._BLOCK_SIZE_D = BLOCK_SIZE_D
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）"""
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 调用 Triton kernel
        _jagged_flash_attention_bwd_preprocess_basic_kernel[self._grid](
            self._o,
            self._o_offset,
            self._do,
            self._delta,
            self._o.stride(0),
            self._o.stride(1),
            self._max_seq_len,
            self._D,
            BLOCK_SIZE_M=self._BLOCK_SIZE_M,
            BLOCK_SIZE_D=self._BLOCK_SIZE_D,
        )

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== NPU dtype 适配（Fix）======
        # 输入曾被强转 fp32；此处把输出转回原始 dtype，与 GPU golden dtype 一致。
        # 原始（未转回）实现：return self._delta
        print('look! -> before') #调试用，随后删
        print(self._delta.dtype) #调试用，随后删
        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and self._delta.dtype != orig_dtype:
            print('look! -> orig_dtype') #调试用，随后删
            print(self._delta.dtype) #调试用，随后删
            print(orig_dtype) #调试用，随后删
            return self._delta.to(orig_dtype)
        return self._delta