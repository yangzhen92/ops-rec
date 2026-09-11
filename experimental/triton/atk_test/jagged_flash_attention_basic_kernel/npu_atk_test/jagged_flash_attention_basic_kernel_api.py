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
"""jagged_flash_attention_basic_kernel 算子的ATK自定义API执行文件（autotune 版）"""

import os
import sys
import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jagged_flash_attention_basic_kernel import (  # noqa: E402
    jagged_flash_attention_basic_kernel,
)


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
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
        seg_len = step + (i % 2)
        seg_len = max(1, seg_len)
        if current + seg_len > total_elements:
            seg_len = total_elements - current
        current += seg_len
        offsets.append(current)
        if current >= total_elements:
            break
    while len(offsets) < expected_length:
        offsets.append(total_elements)
    if offsets[-1] != total_elements:
        offsets[-1] = total_elements
    result = torch.tensor(offsets, dtype=torch.int64, device=device)
    assert result.shape[0] == expected_length
    assert result[0].item() == 0
    assert result[-1].item() == total_elements
    return result


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
        if q_input is None or k_input is None or v_input is None:
            raise ValueError("q, k, v are required")
        if isinstance(q_input, torch.Tensor):
            self._orig_dtype = q_input.dtype
        else:
            self._orig_dtype = torch.float32
        if not isinstance(q_input, torch.Tensor):
            q = torch.tensor(q_input, dtype=torch.float32, device=device)
        else:
            q = q_input.to(device=device, dtype=torch.float32)
        if not isinstance(k_input, torch.Tensor):
            k = torch.tensor(k_input, dtype=torch.float32, device=device)
        else:
            k = k_input.to(device=device, dtype=torch.float32)
        if not isinstance(v_input, torch.Tensor):
            v = torch.tensor(v_input, dtype=torch.float32, device=device)
        else:
            v = v_input.to(device=device, dtype=torch.float32)
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
            if atk_offset is not None:
                offset_shape = list(atk_offset.shape)
            else:
                offset_shape = [9]
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
        o = torch.zeros_like(q)
        lse = torch.empty((total_seq,), device=device, dtype=q.dtype)

        def _grid(META):
            num_blocks_m = triton.cdiv(max_seq_len, META["BLOCK_SIZE_M"])
            return (num_blocks_m, B)

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
        self._grid = _grid
        self._B = B

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        # fmt: off
        jagged_flash_attention_basic_kernel[self._grid](
            q_ptr=self._q, k_ptr=self._k, v_ptr=self._v, offset_ptr=self._offset, o_ptr=self._o, lse_i_ptr=self._lse,
            stride_qm=self._q.stride(0), stride_qd=self._q.stride(1), stride_kd=self._k.stride(0), stride_kn=self._k.stride(1),
            stride_vn=self._v.stride(0), stride_vd=self._v.stride(1), stride_om=self._o.stride(0), stride_od=self._o.stride(1),
            max_seq_len=self._max_seq_len, D=self._D, NEXT_D=self._NEXT_D, use_mask=self._use_mask, allow_tf32=self._allow_tf32,
        )
        # fmt: on
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and self._o.dtype != orig_dtype:
            return self._o.to(orig_dtype)
        return self._o
