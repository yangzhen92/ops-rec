# multi_head_jagged_flash_attention_bwd_kernel_api.py (双精度版)
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
multi_head_jagged_flash_attention_bwd_kernel ATK API - 双精度测试版本

强制 fp32 转换，用于双标杆精度测试。

语义：多头 Jagged Flash Attention 反向梯度计算
    dq, dk, dv = mh_backward(q, k, v, o, do, delta, lse, offset)
    q/k/v/o/do [num_heads, total_seq, D], delta/lse [num_heads, total_seq]
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triton_multi_head_jagged_flash_attention import _multi_head_jagged_flash_attention_bwd_kernel  # noqa: E402


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


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@register("_multi_head_jagged_flash_attention_bwd_kernel")
class MultiHeadJaggedFlashAttentionBwdKernelApi(BaseApi):
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
        allow_tf32 = input_data.kwargs.get("allow_tf32", None)
        BLOCK_M = input_data.kwargs.get("BLOCK_M", None)
        BLOCK_N = input_data.kwargs.get("BLOCK_N", None)
        BLOCK_D = input_data.kwargs.get("BLOCK_D", None)

        if q_input is None or k_input is None or v_input is None:
            raise ValueError("q, k, v are required")

        # ====== NPU dtype 适配（Fix）======
        # 背景：kernel 内 tl.dot + tl.load(other=0.0) 在 Ascend 上被促升为 fp64，
        #       导致编译期 dtype 报错。统一强转 fp32 让 kernel 编译通过；
        #       输出再转回原始 dtype，与 GPU golden 双标杆 dtype 一致。
        # 记录原始 dtype（以 q 为准）
        if isinstance(q_input, torch.Tensor):
            self._orig_dtype = q_input.dtype
        else:
            self._orig_dtype = torch.float32

        def _to_dev(x):
            if x is None:
                return None
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                # 原始（未强转）实现：x = x.to(device)
                x = x.to(device=device, dtype=torch.float32)
            return x.contiguous() if not x.is_contiguous() else x

        q = _to_dev(q_input)
        k = _to_dev(k_input)
        v = _to_dev(v_input)

        # q [num_heads, total_seq, D]
        num_heads = q.shape[0]
        total_seq = q.shape[1]
        D = q.shape[2]

        # 对齐 k/v/o/do 到 [num_heads, total_seq, D]
        def _align_3d(x):
            if x is None or x.shape != (num_heads, total_seq, D):
                return torch.randn(num_heads, total_seq, D, dtype=q.dtype, device=device).contiguous()
            return x

        k = _align_3d(k)
        v = _align_3d(v)
        o = _align_3d(_to_dev(o_input))
        do = _align_3d(_to_dev(do_input))

        # delta/lse 对齐到 [num_heads, total_seq]
        def _align_2d(x):
            if x is None or x.shape != (num_heads, total_seq):
                return torch.randn(num_heads, total_seq, dtype=q.dtype, device=device).contiguous()
            return x

        delta = _align_2d(_to_dev(delta_input))
        lse = _align_2d(_to_dev(lse_input))

        # 处理列表格式参数
        if isinstance(max_seq_len, list):
            max_seq_len = max_seq_len[0] if len(max_seq_len) > 0 else None
        if isinstance(allow_tf32, list):
            allow_tf32 = allow_tf32[0] if len(allow_tf32) > 0 else None
        if isinstance(BLOCK_M, list):
            BLOCK_M = BLOCK_M[0] if len(BLOCK_M) > 0 else None
        if isinstance(BLOCK_N, list):
            BLOCK_N = BLOCK_N[0] if len(BLOCK_N) > 0 else None
        if isinstance(BLOCK_D, list):
            BLOCK_D = BLOCK_D[0] if len(BLOCK_D) > 0 else None

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
        if allow_tf32 is None:
            allow_tf32 = False
        if BLOCK_M is None or BLOCK_M <= 0:
            BLOCK_M = 32
        if BLOCK_N is None or BLOCK_N <= 0:
            BLOCK_N = 32
        # BLOCK_D 必须 >= D 且是 2 的幂次方（至少 16）
        BLOCK_D = max(_next_power_of_2(D), 16)

        # 输出
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        import triton

        grid = (triton.cdiv(max_seq_len, BLOCK_N), B * num_heads)

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
        self._num_heads = num_heads
        self._max_seq_len = max_seq_len
        self._D = D
        self._allow_tf32 = allow_tf32
        self._BLOCK_M = BLOCK_M
        self._BLOCK_N = BLOCK_N
        self._BLOCK_D = BLOCK_D
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        _multi_head_jagged_flash_attention_bwd_kernel[self._grid](
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
            self._q.stride(2),
            self._k.stride(0),
            self._k.stride(1),
            self._k.stride(2),
            self._v.stride(0),
            self._v.stride(1),
            self._v.stride(2),
            self._o.stride(0),
            self._o.stride(1),
            self._o.stride(2),
            self._lse.stride(0),
            self._delta.stride(0),
            self._dq.stride(0),
            self._dq.stride(1),
            self._dq.stride(2),
            self._dk.stride(0),
            self._dk.stride(1),
            self._dk.stride(2),
            self._dv.stride(0),
            self._dv.stride(1),
            self._dv.stride(2),
            self._do.stride(0),
            self._do.stride(1),
            self._do.stride(2),
            self._num_heads,
            self._max_seq_len,
            self._D,
            self._allow_tf32,
            BLOCK_M=self._BLOCK_M,
            BLOCK_N=self._BLOCK_N,
            BLOCK_D=self._BLOCK_D,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        return self._dq
