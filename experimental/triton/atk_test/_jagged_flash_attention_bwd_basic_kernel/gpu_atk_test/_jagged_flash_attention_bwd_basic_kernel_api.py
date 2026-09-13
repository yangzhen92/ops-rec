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
jagged_flash_attention_bwd_basic_kernel ATK API

语义：Flash Attention 反向梯度计算
    dq, dk, dv = backward(q, k, v, o, do, delta, lse, offset)
    q/k/v/o/do [total_seq, D], delta/lse [total_seq]
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_jagged_flash_attention_basic import _jagged_flash_attention_bwd_basic_kernel


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:
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


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@register("_jagged_flash_attention_bwd_basic_kernel")
class JaggedFlashAttentionBwdBasicKernelApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        q_input = input_data.kwargs.get("q", None)
        k_input = input_data.kwargs.get("k", None)
        v_input = input_data.kwargs.get("v", None)
        o_input = input_data.kwargs.get("o", None)
        do_input = input_data.kwargs.get("do", None)
        delta_input = input_data.kwargs.get("delta", None)
        lse_input = input_data.kwargs.get("lse", None)
        offset_input = input_data.kwargs.get("offset", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", None)
        use_mask = input_data.kwargs.get("use_mask", None)
        allow_tf32 = input_data.kwargs.get("allow_tf32", None)
        BLOCK_SIZE_M = input_data.kwargs.get("BLOCK_SIZE_M", None)
        BLOCK_SIZE_N = input_data.kwargs.get("BLOCK_SIZE_N", None)
        BLOCK_SIZE_D = input_data.kwargs.get("BLOCK_SIZE_D", None)

        if q_input is None or k_input is None or v_input is None:
            raise ValueError("q, k, v are required")

        # 原生版：不强转精度，保持输入 dtype
        # 原生版：不强转精度，保持输入 dtype
        def _to_dev(x):
            if x is None:
                return None
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=device)
            else:
                # 原始（未强转）实现：x = x.to(device)
                x = x.to(device=device)
            return x.contiguous() if not x.is_contiguous() else x

        q = _to_dev(q_input)
        k = _to_dev(k_input)
        v = _to_dev(v_input)

        # 以 q 为基准 [total_seq, D]
        total_seq = q.shape[0]
        D = q.shape[1]

        # 对齐 k/v/o/do 到 [total_seq, D]
        def _align_2d(x):
            if x is None or x.shape != (total_seq, D):
                return torch.randn(total_seq, D, dtype=q.dtype, device=device).contiguous()
            return x

        k = _align_2d(k)
        v = _align_2d(v)
        o = _align_2d(_to_dev(o_input))
        do = _align_2d(_to_dev(do_input))

        # delta/lse 对齐到 [total_seq]
        def _align_1d(x):
            if x is None or x.shape != (total_seq,):
                return torch.randn(total_seq, dtype=q.dtype, device=device).contiguous()
            return x

        delta = _align_1d(_to_dev(delta_input))
        lse = _align_1d(_to_dev(lse_input))

        # 处理列表格式参数
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

        # 生成或处理 offset
        if offset_input is None:
            offset = _generate_jagged_offset_by_shape([9], total_seq, device)
        elif isinstance(offset_input, torch.Tensor):
            offset = _generate_jagged_offset_by_shape(list(offset_input.shape), total_seq, device)
        elif isinstance(offset_input, list):
            if len(offset_input) > 0 and hasattr(offset_input[0], 'shape'):
                offset_shape = list(offset_input[0].shape)
            else:
                offset_shape = [9]
            offset = _generate_jagged_offset_by_shape(offset_shape, total_seq, device)
        else:
            offset = _generate_jagged_offset_by_shape([9], total_seq, device)

        B = offset.shape[0] - 1

        # 默认值
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

        # 关键约束：BLOCK_SIZE_D 必须 >= D（kernel 一次性处理所有 D）
        if BLOCK_SIZE_D < D:
            BLOCK_SIZE_D = _next_power_of_2(D)
            # 当 D >= 128 时，降低 BLOCK_M/N 避免 shared memory 溢出
            # A5 硬件限制：L0A/B = 64KB，三个大 BLOCK 会超出 101376 字节
            if D >= 128:
                BLOCK_SIZE_M = min(BLOCK_SIZE_M, 16)
                BLOCK_SIZE_N = min(BLOCK_SIZE_N, 16)

        # 输出
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        import triton

        grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_N), B)

        self._device = device
        self._q = q
        self._k = k
        self._v = v
        self._o = o
        self._do = do
        self._delta = delta
        self._lse = lse
        self._offset = offset
        self._dq = dq
        self._dk = dk
        self._dv = dv
        self._max_seq_len = max_seq_len
        self._D = D
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

        _jagged_flash_attention_bwd_basic_kernel[self._grid](
            self._q,
            self._k,
            self._v,
            self._o,
            self._offset,
            self._dq,
            self._dk,
            self._dv,
            self._do,
            self._delta,
            self._lse,
            self._q.stride(0),
            self._q.stride(1),
            self._k.stride(0),
            self._k.stride(1),
            self._v.stride(0),
            self._v.stride(1),
            self._o.stride(0),
            self._o.stride(1),
            self._dq.stride(0),
            self._dq.stride(1),
            self._dk.stride(0),
            self._dk.stride(1),
            self._dv.stride(0),
            self._dv.stride(1),
            self._do.stride(0),
            self._do.stride(1),
            self._max_seq_len,
            self._D,
            self._use_mask,
            self._allow_tf32,
            BLOCK_SIZE_M=self._BLOCK_SIZE_M,
            BLOCK_SIZE_N=self._BLOCK_SIZE_N,
            BLOCK_SIZE_D=self._BLOCK_SIZE_D,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== NPU dtype 适配（Fix）======
        # 输入曾被强转 fp32；此处把输出转回原始 dtype，与 GPU golden dtype 一致。
        # 原始（未转回）实现：return self._dq
        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and self._dq.dtype != orig_dtype:
            return self._dq.to(orig_dtype)
        return self._dq
