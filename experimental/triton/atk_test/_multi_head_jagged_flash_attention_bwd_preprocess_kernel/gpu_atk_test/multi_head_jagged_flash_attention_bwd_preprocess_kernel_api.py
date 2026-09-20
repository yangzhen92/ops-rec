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
# limitations under the License.
# ==============================================================================

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_multi_head_jagged_flash_attention import _multi_head_jagged_flash_attention_bwd_preprocess_kernel


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

    assert result.shape[0] == expected_length, f"offset length mismatch: {result.shape[0]} != {expected_length}"
    assert result[0].item() == 0, f"offset[0] should be 0, got {result[0].item()}"
    assert result[-1].item() == total_elements, f"offset[-1] should be {total_elements}, got {result[-1].item()}"

    return result


@register("_multi_head_jagged_flash_attention_bwd_preprocess_kernel")
class MultiHeadJaggedFlashAttentionBwdPreprocessKernelApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        # pylint: disable=attribute-defined-outside-init
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        o_input = input_data.kwargs.get("o", None)
        do_input = input_data.kwargs.get("do", None)
        o_offset_input = input_data.kwargs.get("o_offset", None)
        num_heads = input_data.kwargs.get("num_heads", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", None)
        BLOCK_M = input_data.kwargs.get("BLOCK_M", None)
        BLOCK_D = input_data.kwargs.get("BLOCK_D", None)

        if o_input is None or do_input is None:
            raise ValueError("o and do are required")

        if isinstance(o_input, torch.Tensor):
            self._orig_dtype = o_input.dtype
        else:
            self._orig_dtype = torch.float32

        needs_cast = self._orig_dtype == torch.float64

        def _to_device(t):
            if not isinstance(t, torch.Tensor):
                return torch.tensor(t, dtype=(torch.float32 if needs_cast else None), device=device)
            return t.to(device=device, dtype=(torch.float32 if needs_cast else None))

        o = _to_device(o_input)
        do = _to_device(do_input)

        if not o.is_contiguous():
            o = o.contiguous()
        if not do.is_contiguous():
            do = do.contiguous()

        if len(o.shape) != 3:
            raise ValueError(f"o should be 3D tensor, got shape {o.shape}")

        num_heads_from_shape = o.shape[0]
        total_seq = o.shape[1]
        D = o.shape[2]

        if num_heads != num_heads_from_shape:
            num_heads = num_heads_from_shape

        if o.shape != do.shape:
            raise ValueError(f"o.shape {o.shape} != do.shape {do.shape}")

        if o_offset_input is None:
            B = 8
            o_offset = _generate_jagged_offset_by_shape([B + 1], total_seq, device)
        else:
            if isinstance(o_offset_input, list):
                if len(o_offset_input) > 0 and hasattr(o_offset_input[0], 'shape'):
                    offset_shape = list(o_offset_input[0].shape)
                else:
                    offset_shape = [9]  # 默认 B=8
                o_offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
            elif isinstance(o_offset_input, torch.Tensor):
                o_offset = o_offset_input.to(device)
                if o_offset[-1].item() > total_seq:
                    o_offset = _generate_jagged_offset_by_shape(list(o_offset.shape), total_seq, device)
            else:
                B = 8
                o_offset = _generate_jagged_offset_by_shape([B + 1], total_seq, device)

        B = o_offset.shape[0] - 1

        if isinstance(num_heads, list):
            num_heads = num_heads[0] if len(num_heads) > 0 else None
        if isinstance(max_seq_len, list):
            max_seq_len = max_seq_len[0] if len(max_seq_len) > 0 else None
        if isinstance(BLOCK_M, list):
            BLOCK_M = BLOCK_M[0] if len(BLOCK_M) > 0 else None
        if isinstance(BLOCK_D, list):
            BLOCK_D = BLOCK_D[0] if len(BLOCK_D) > 0 else None

        if num_heads is None or num_heads <= 0:
            num_heads = num_heads_from_shape

        if max_seq_len is None or max_seq_len <= 0:
            if B > 0:
                avg_seq = (total_seq + B - 1) // B
                max_seq_len = 1
                while max_seq_len < avg_seq:
                    max_seq_len *= 2
                max_seq_len = max(16, max_seq_len)
            else:
                max_seq_len = 128

        if BLOCK_M is None or BLOCK_M <= 0:
            BLOCK_M = 32

        if BLOCK_D is None or BLOCK_D <= 0:
            BLOCK_D = 64

        if BLOCK_D < D:
            BLOCK_D = 1
            while BLOCK_D < D:
                BLOCK_D *= 2

        if isinstance(o_offset, torch.Tensor) and o_offset.shape[0] > 1:
            offset_shape = list(o_offset.shape)
            o_offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
            B = o_offset.shape[0] - 1

        delta = torch.zeros(num_heads, total_seq, dtype=torch.float32, device=device)

        import triton

        grid = (
            triton.cdiv(max_seq_len, BLOCK_M),
            B * num_heads,
        )

        self._device = device
        self._o = o
        self._do = do
        self._o_offset = o_offset
        self._delta = delta
        self._num_heads = num_heads
        self._max_seq_len = max_seq_len
        self._D = D
        self._BLOCK_M = BLOCK_M
        self._BLOCK_D = BLOCK_D
        self._grid = grid
        # pylint: enable=attribute-defined-outside-init

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        _multi_head_jagged_flash_attention_bwd_preprocess_kernel[self._grid](
            self._o,
            self._o_offset,
            self._do,
            self._delta,
            self._o.stride(0),
            self._o.stride(1),
            self._o.stride(2),
            self._delta.stride(0),
            num_heads=self._num_heads,
            max_seq_len=self._max_seq_len,
            D=self._D,
            BLOCK_M=self._BLOCK_M,
            BLOCK_D=self._BLOCK_D,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and self._delta.dtype != orig_dtype:
            return self._delta.to(orig_dtype)
        return self._delta
