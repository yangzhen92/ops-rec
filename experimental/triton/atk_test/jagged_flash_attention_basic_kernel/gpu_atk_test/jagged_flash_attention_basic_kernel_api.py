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
"""jagged_flash_attention_basic_kernel 算子的ATK自定义API执行文件（GPU 原生版）"""

import os
import sys
import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jagged_flash_attention_basic_kernel import jagged_flash_attention_basic_kernel  # noqa: E402


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _generate_jagged_offset_by_shape(shape, total_elements, device):
    expected_length = shape[0]
    B = expected_length - 1
    if B <= 0:
        return torch.tensor([0, total_elements], dtype=torch.int64, device=device)
    offsets = [0]
    step = total_elements // B
    current = 0
    for i in range(B):
        seg_len = max(1, step + (i % 2))
        if current + seg_len > total_elements:
            seg_len = total_elements - current
        current += seg_len
        offsets.append(current)
        if current >= total_elements:
            break
    while len(offsets) < expected_length:
        offsets.append(total_elements)
    offsets[-1] = total_elements
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _is_offset_valid(offset, total_seq):
    if offset is None or offset.numel() <= 1:
        return False
    if int(offset[0].item()) != 0:
        return False
    if int(offset[-1].item()) != total_seq:
        return False
    for i in range(offset.numel() - 1):
        if int(offset[i + 1].item()) < int(offset[i].item()):
            return False
    return True


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@register("jagged_flash_attention_basic_kernel")
class JaggedFlashAttentionBasicKernelApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        q_input = input_data.kwargs.get("q", None)
        k_input = input_data.kwargs.get("k", None)
        v_input = input_data.kwargs.get("v", None)
        offset_input = input_data.kwargs.get("offset", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", None)
        use_mask = input_data.kwargs.get("use_mask", None)
        allow_tf32 = input_data.kwargs.get("allow_tf32", None)
        BLOCK_SIZE_M = input_data.kwargs.get("BLOCK_SIZE_M", None)
        BLOCK_SIZE_N = input_data.kwargs.get("BLOCK_SIZE_N", None)
        BLOCK_SIZE_D = input_data.kwargs.get("BLOCK_SIZE_D", None)
        if q_input is None or k_input is None or v_input is None:
            raise ValueError("q, k, v are required")
        if not isinstance(q_input, torch.Tensor):
            q = torch.tensor(q_input, device=device)
        else:
            q = q_input.to(device=device)
        if not isinstance(k_input, torch.Tensor):
            k = torch.tensor(k_input, device=device)
        else:
            k = k_input.to(device=device)
        if not isinstance(v_input, torch.Tensor):
            v = torch.tensor(v_input, device=device)
        else:
            v = v_input.to(device=device)
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        total_seq = q.shape[0]
        D = q.shape[1]
        if k.shape != (D, total_seq):
            k = torch.zeros(D, total_seq, dtype=q.dtype, device=device).contiguous()
        if v.shape != (total_seq, D):
            v = torch.zeros(total_seq, D, dtype=q.dtype, device=device).contiguous()
        if isinstance(max_seq_len, list):
            max_seq_len = max_seq_len[0] if len(max_seq_len) > 0 else None
        if isinstance(use_mask, list):
            use_mask = use_mask[0] if len(use_mask) > 0 else None
        if isinstance(allow_tf32, list):
            allow_tf32 = allow_tf32[0] if len(allow_tf32) > 0 else None
        if isinstance(BLOCK_SIZE_M, list):
            BLOCK_SIZE_M = BLOCK_SIZE_M[0] if len(BLOCK_SIZE_M) > 0 else None
        if isinstance(BLOCK_SIZE_N, list):
            BLOCK_SIZE_N = BLOCK_SIZE_N[0] if len(BLOCK_SIZE_N) > 0 else None
        if isinstance(BLOCK_SIZE_D, list):
            BLOCK_SIZE_D = BLOCK_SIZE_D[0] if len(BLOCK_SIZE_D) > 0 else None
        atk_offset = None
        if isinstance(offset_input, torch.Tensor):
            atk_offset = offset_input.to(device=device, dtype=torch.int64).contiguous()
        elif isinstance(offset_input, list):
            if len(offset_input) > 0 and hasattr(offset_input[0], 'shape'):
                atk_offset = offset_input[0].to(device=device, dtype=torch.int64).contiguous()
            else:
                try:
                    atk_offset = torch.tensor(offset_input, dtype=torch.int64, device=device)
                except Exception:
                    atk_offset = None
        if atk_offset is not None and _is_offset_valid(atk_offset, total_seq):
            offset = atk_offset
        else:
            offset_shape = list(atk_offset.shape) if atk_offset is not None else [9]
            offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
        B = offset.shape[0] - 1
        if max_seq_len is None or max_seq_len <= 0:
            if B > 0:
                avg_seq = (total_seq + B - 1) // B
                max_seq_len = max(16, _next_power_of_2(avg_seq))
            else:
                max_seq_len = 128
        if use_mask is None:
            use_mask = False
        if allow_tf32 is None:
            allow_tf32 = False
        if BLOCK_SIZE_M is None or BLOCK_SIZE_M <= 0:
            BLOCK_SIZE_M = 32
        if BLOCK_SIZE_N is None or BLOCK_SIZE_N <= 0:
            BLOCK_SIZE_N = 32
        if BLOCK_SIZE_D is None or BLOCK_SIZE_D <= 0:
            BLOCK_SIZE_D = 32
        o = torch.zeros_like(q)
        lse = torch.empty((total_seq,), device=device, dtype=q.dtype)
        grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_M), B)
        self._device = device
        self._q = q
        self._k = k
        self._v = v
        self._offset = offset
        self._o = o
        self._lse = lse
        self._max_seq_len = max_seq_len
        self._D = D
        self._NEXT_D = _next_power_of_2(D)
        self._use_mask = use_mask
        self._allow_tf32 = allow_tf32
        self._BLOCK_SIZE_M = BLOCK_SIZE_M
        self._BLOCK_SIZE_N = BLOCK_SIZE_N
        self._BLOCK_SIZE_D = BLOCK_SIZE_D
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        # fmt: off
        jagged_flash_attention_basic_kernel[self._grid](
            self._q, self._k, self._v, self._offset, self._o, self._lse, self._q.stride(0), self._q.stride(1),
            self._k.stride(0), self._k.stride(1), self._v.stride(0), self._v.stride(1), self._o.stride(0), self._o.stride(1),
            self._max_seq_len, self._D, self._NEXT_D, self._use_mask, self._allow_tf32, BLOCK_SIZE_M=self._BLOCK_SIZE_M,
            BLOCK_SIZE_N=self._BLOCK_SIZE_N, BLOCK_SIZE_D=self._BLOCK_SIZE_D,
        )
        # fmt: on
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        return self._o
