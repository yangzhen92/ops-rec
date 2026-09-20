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

# triton_jagged_jagged_bmm_reduce_sum_api.py
"""_jagged_jagged_bmm_reduce_sum 的 NPU autotune ATK API 封装。"""

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_jagged_jagged_bmm_reduce_sum_kernel import _jagged_jagged_bmm_reduce_sum


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MAX_TOTAL_ELEMENTS = 1 << 24


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_jagged_jagged_bmm_reduce_sum requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _build_offsets(batch_size: int, total_length: int, device: str) -> tuple[torch.Tensor, list[int]]:
    """生成严格递增、非均匀的 offsets，每段至少一个 token。"""
    if batch_size < 1 or total_length < batch_size:
        raise ValueError(f"Require L >= B >= 1, got L={total_length}, B={batch_size}")

    remaining = total_length - batch_size
    weights = [1 + ((batch_index * 5 + total_length) % 7) for batch_index in range(batch_size)]
    weight_sum = sum(weights)
    extra_lengths = [remaining * weight // weight_sum for weight in weights]
    residue = remaining - sum(extra_lengths)
    for residue_index in range(residue):
        extra_lengths[(residue_index * 3 + total_length) % batch_size] += 1

    offsets = [0]
    seg_lens = []
    for batch_index in range(batch_size):
        segment_length = 1 + extra_lengths[batch_index]
        offsets.append(offsets[-1] + segment_length)
        seg_lens.append(segment_length)

    assert offsets[0] == 0
    assert offsets[-1] == total_length
    assert all(right > left for left, right in zip(offsets, offsets[1:]))
    return torch.tensor(offsets, dtype=torch.int64, device=device), seg_lens


@register("autotune_triton_jagged_jagged_bmm_reduce_sum")
@register("triton_jagged_jagged_bmm_reduce_sum")
@register("triton_jagged._jagged_jagged_bmm_reduce_sum")
@register("_jagged_jagged_bmm_reduce_sum")
class TritonJaggedJaggedBmmReduceSumPerfApi(BaseApi):
    """_jagged_jagged_bmm_reduce_sum 的 Autotune API 封装。"""

    def __init__(self):
        self._device = self._jagged_a = self._jagged_b = None
        self._out = self._reduce_out = self._seq_offsets = None
        self._grid = None
        self._m = self._n = self._autotune_max_seq_len = 0

    def init_by_input_data(self, input_data: InputDataset):
        jagged_a_input = input_data.kwargs.get("JaggedA")
        jagged_b_input = input_data.kwargs.get("JaggedB")
        reduce_out_input = input_data.kwargs.get("ReduceOut")
        offsets_input = input_data.kwargs.get("seq_offsets")

        if not all(
            isinstance(v, torch.Tensor) for v in (jagged_a_input, jagged_b_input, reduce_out_input, offsets_input)
        ):
            raise ValueError("JaggedA, JaggedB, ReduceOut and seq_offsets must all be tensors")
        if jagged_a_input.ndim != 2 or jagged_b_input.ndim != 2 or reduce_out_input.ndim != 2:
            raise ValueError("Expected JaggedA[L,M], JaggedB[L,N], ReduceOut[B,N]")
        if offsets_input.ndim != 1 or offsets_input.numel() < 2 or offsets_input.dtype != torch.int64:
            raise ValueError("seq_offsets must be an int64 tensor with shape [B + 1]")

        # ATK CPU golden 可能将 fp32 升为 fp64，而 tl.dot 不支持 fp64。
        if jagged_a_input.dtype == torch.float64:
            jagged_a_input = jagged_a_input.to(torch.float32)
            jagged_b_input = jagged_b_input.to(torch.float32)
            reduce_out_input = reduce_out_input.to(torch.float32)

        if jagged_a_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {jagged_a_input.dtype}")
        if jagged_b_input.dtype != jagged_a_input.dtype or reduce_out_input.dtype != jagged_a_input.dtype:
            raise ValueError("Generator must enforce one shared dtype for all inputs")

        l_a, m = jagged_a_input.shape
        l_b, n = jagged_b_input.shape
        if l_a != l_b:
            raise ValueError(f"JaggedA and JaggedB must have same L, got {l_a} vs {l_b}")
        total_length = l_a
        batch_size = offsets_input.numel() - 1
        if batch_size < 1 or total_length < batch_size:
            raise ValueError(f"Require L >= B >= 1, got L={total_length}, B={batch_size}")
        if tuple(reduce_out_input.shape) != (batch_size, n):
            raise ValueError(f"ReduceOut must have shape [{batch_size}, {n}], got {tuple(reduce_out_input.shape)}")

        total_elements = total_length * m + total_length * n + batch_size * m * n + batch_size * n
        if total_elements > _MAX_TOTAL_ELEMENTS:
            raise ValueError("Generator must enforce total-element memory limit")

        device = _get_device()
        self._device = device

        self._jagged_a = jagged_a_input.to(device).contiguous()
        self._jagged_b = jagged_b_input.to(device).contiguous()
        self._out = torch.empty((batch_size, m, n), dtype=self._jagged_a.dtype, device=device)
        self._reduce_out = torch.empty((batch_size, n), dtype=self._jagged_a.dtype, device=device)
        self._seq_offsets, seg_lens = _build_offsets(batch_size, total_length, device)

        self._m = m
        self._n = n
        self._autotune_max_seq_len = max(seg_lens)

        self._grid = lambda meta: (
            batch_size,
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        _jagged_jagged_bmm_reduce_sum[self._grid](
            seq_offsets=self._seq_offsets,
            JaggedA=self._jagged_a,
            JaggedB=self._jagged_b,
            Out=self._out,
            ReduceOut=self._reduce_out,
            M=self._m,
            N=self._n,
            AUTOTUNE_MAX_SEQ_LEN=self._autotune_max_seq_len,
            stride_ak=self._jagged_a.stride(0),
            stride_bk=self._jagged_b.stride(0),
            stride_ob=self._out.stride(0),
            stride_om=self._out.stride(1),
            stride_on=self._out.stride(2),
            stride_orb=self._reduce_out.stride(0),
            stride_orn=self._reduce_out.stride(1),
            REDUCE_JAGGEDB=False,
            ALLOW_TF32=False,
        )

        _synchronize(self._device)

        return self._out
