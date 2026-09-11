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
# pylint: disable=attribute-defined-outside-init
"""
jagged_dense_elementwise_mul_jagged_out_kernel 算子的ATK自定义API执行文件

语义：jagged_A 每 batch 段为 seq² 方阵展平，dense_B 为 [max_seq_len, max_seq_len] 共享方阵，
逐元素相乘输出到 jagged_C；grid = (B, cdiv(max_seq_len, BLOCK_M))
"""

import os
import sys

import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jagged_dense_elementwise_mul_jagged_out_kernel import (  # noqa: E402
    jagged_dense_elementwise_mul_jagged_out_kernel,
)


def get_device():
    """检测可用设备（NPU优先）"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@register("jagged_dense_elementwise_mul_jagged_out_kernel")
class JaggedDenseElementwiseMulJaggedOutKernelAPI(BaseApi):
    """jagged_dense_elementwise_mul_jagged_out_kernel 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()

        a_ptr_input = input_data.kwargs.get("a_ptr", None)
        b_ptr_input = input_data.kwargs.get("b_ptr", None)
        a_offsets_ptr_input = input_data.kwargs.get("a_offsets_ptr", None)

        if a_ptr_input is None or b_ptr_input is None or a_offsets_ptr_input is None:
            raise ValueError("a_ptr, b_ptr or a_offsets_ptr not found")

        # NPU dtype 适配：记录原始dtype
        if isinstance(a_ptr_input, torch.Tensor):
            self._orig_dtype = a_ptr_input.dtype
        else:
            self._orig_dtype = torch.float32

        # 强转fp32避免NPU编译问题
        a_ptr_raw = self._to_tensor(a_ptr_input, device, dtype=torch.float32)
        b_ptr_raw = self._to_tensor(b_ptr_input, device, dtype=torch.float32)
        a_offsets_ptr_raw = self._to_tensor(a_offsets_ptr_input, device, torch.int32)

        B = a_offsets_ptr_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        dtype = a_ptr_raw.dtype

        # 从 offsets（平方 cumsum）反推 seq_lengths；offset 非法时用均分策略兜底
        a_offsets_ptr = a_offsets_ptr_raw.contiguous()
        offsets_valid = True
        for i in range(B):
            if a_offsets_ptr[i + 1].item() < a_offsets_ptr[i].item():
                offsets_valid = False
                break

        if offsets_valid:
            lengths_seq = []
            for i in range(B):
                square_seg_len = int(a_offsets_ptr[i + 1].item() - a_offsets_ptr[i].item())
                square_seg_len = max(square_seg_len, 0)
                seg_len = int(square_seg_len**0.5)
                lengths_seq.append(seg_len)
        else:
            total_seq = a_ptr_raw.shape[0]
            avg_square_seg = total_seq // B
            avg_seg_len = int(avg_square_seg**0.5)
            lengths_seq = [avg_seg_len] * B
            new_offsets = [0]
            for seg_len in lengths_seq:
                new_offsets.append(new_offsets[-1] + seg_len * seg_len)
            a_offsets_ptr = torch.tensor(new_offsets, dtype=torch.int32, device=device)

        # total_seq 用实际输入长度；max_seq_len 从 b_ptr 方阵 shape 推断（至少 1）
        total_seq = a_ptr_raw.shape[0]
        if len(b_ptr_raw.shape) >= 2:
            max_seq_len = max(b_ptr_raw.shape[0], b_ptr_raw.shape[1])
        else:
            max_seq_len = max(lengths_seq) if lengths_seq else 1
        max_seq_len = max(1, max_seq_len)

        a_seq_lengths_ptr = torch.tensor(lengths_seq, dtype=torch.int32, device=device)
        a_ptr = self._rebuild_1d(a_ptr_raw, total_seq, dtype, device)
        b_ptr = self._rebuild_2d(b_ptr_raw, max_seq_len, max_seq_len, dtype, device)
        c_ptr = torch.zeros((total_seq,), dtype=dtype, device=device)

        BLOCK_M = 32
        grid = (B, triton.cdiv(max_seq_len, BLOCK_M))

        self._device = device
        self._a_ptr = a_ptr
        self._b_ptr = b_ptr
        self._c_ptr = c_ptr
        self._a_seq_lengths_ptr = a_seq_lengths_ptr
        self._a_offsets_ptr = a_offsets_ptr
        self._max_seq_len = max_seq_len
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        BLOCK_M = 32
        BLOCK_N = 32

        # fmt: off
        jagged_dense_elementwise_mul_jagged_out_kernel[self._grid](
            self._a_ptr, self._b_ptr, self._c_ptr, self._a_seq_lengths_ptr,
            self._a_offsets_ptr, self._a_ptr.stride(0),
            self._b_ptr.stride(0), self._b_ptr.stride(1), self._max_seq_len,
            BLOCK_M, BLOCK_N,
        )
        # fmt: on

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 输出转回原始 dtype，与 GPU golden 保持一致
        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and self._c_ptr.dtype != orig_dtype:
            return self._c_ptr.to(orig_dtype)
        return self._c_ptr

    def _rebuild_2d(self, raw, rows, cols, dtype, device):
        """重建2D tensor"""
        target_len = rows * cols
        flat = self._rebuild_1d(raw, target_len, dtype, device)
        return flat.reshape(rows, cols).contiguous()

    def _rebuild_1d(self, raw, target_len, dtype, device):
        """重建 1D tensor 到 target_len（不够时用 pad 补齐，避免 stride 问题）"""
        raw_flat = raw.reshape(-1).to(device=device, dtype=dtype)
        if raw_flat.shape[0] >= target_len:
            return raw_flat[:target_len].contiguous()
        pad_len = target_len - raw_flat.shape[0]
        return torch.nn.functional.pad(raw_flat, (0, pad_len), mode='constant', value=0).contiguous()

    def _to_tensor(self, data, device, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return torch.tensor(data, dtype=dtype, device=device)
        return torch.tensor(data, device=device)
