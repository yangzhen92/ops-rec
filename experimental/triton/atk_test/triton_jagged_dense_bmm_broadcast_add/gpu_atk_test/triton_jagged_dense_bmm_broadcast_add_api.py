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

# triton_jagged_dense_bmm_broadcast_add_api.py
"""jagged_dense_bmm_broadcast_add_kernel 的 ATK API 封装。"""

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_jagged_dense_bmm_broadcast_add_kernel import jagged_dense_bmm_broadcast_add_kernel


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MAX_TOTAL_ELEMENTS = 1 << 24


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("jagged_dense_bmm_broadcast_add_kernel requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _build_offsets(batch_size: int, total_length: int, device: str) -> tuple[torch.Tensor, int]:
    """生成严格递增、非均匀的确定性 offsets；每段至少一个 token。
    返回 (offsets_tensor, max_seq_len)。
    """
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
    max_seq_len = 0
    for batch_index in range(batch_size):
        segment_length = 1 + extra_lengths[batch_index]
        offsets.append(offsets[-1] + segment_length)
        max_seq_len = max(max_seq_len, segment_length)

    assert offsets[0] == 0
    assert offsets[-1] == total_length
    assert all(right > left for left, right in zip(offsets, offsets[1:]))
    return torch.tensor(offsets, dtype=torch.int64, device=device), max_seq_len


@register("triton_jagged.jagged_dense_bmm_broadcast_add_kernel")
@register("triton_jagged_dense_bmm_broadcast_add")
class TritonJaggedDenseBmmBroadcastAddApi(BaseApi):
    """jagged_dense_bmm_broadcast_add_kernel 的 Autotune API 封装。"""

    def __init__(self):
        self._device = None
        self._jagged = None
        self._dense = None
        self._bias = None
        self._seq_offsets = None
        self._max_seq_len = 0
        self._out = None
        self._k = 0
        self._n = 0
        self._grid = None

    def init_by_input_data(self, input_data: InputDataset):
        """根据 ATK InputDataset 组装 kernel 入参（不计时）。"""
        jagged_input = input_data.kwargs.get("Jagged")
        dense_input = input_data.kwargs.get("Dense")
        bias_input = input_data.kwargs.get("Bias")
        offsets_input = input_data.kwargs.get("seq_offsets")

        if not all(isinstance(value, torch.Tensor) for value in (jagged_input, dense_input, bias_input, offsets_input)):
            raise ValueError("Jagged, Dense, Bias and seq_offsets must all be tensors")
        if jagged_input.ndim != 2 or dense_input.ndim != 3 or bias_input.ndim != 2:
            raise ValueError("Expected Jagged[L,K], Dense[B,K,N], and Bias[B,N]")
        if offsets_input.ndim != 1 or offsets_input.numel() < 2 or offsets_input.dtype != torch.int64:
            raise ValueError("seq_offsets must be an int64 tensor with shape [B + 1]")

        # 双标杆下 ATK CPU golden 会将 fp32 升成 fp64；kernel 与 NPU tl.dot
        # 只支持 fp32/fp16/bf16，需先降回 fp32。
        if jagged_input.dtype == torch.float64:
            jagged_input = jagged_input.to(torch.float32)
            dense_input = dense_input.to(torch.float32)
            bias_input = bias_input.to(torch.float32)

        if jagged_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported floating dtype: {jagged_input.dtype}")
        if dense_input.dtype != jagged_input.dtype or bias_input.dtype != jagged_input.dtype:
            raise ValueError("Generator must enforce one shared dtype for Jagged, Dense and Bias")

        total_length, k = jagged_input.shape
        batch_size, dense_k, n = dense_input.shape
        if dense_k != k or tuple(bias_input.shape) != (batch_size, n):
            raise ValueError("Generator must enforce Dense[B,K,N] and Bias[B,N]")
        if offsets_input.numel() != batch_size + 1 or total_length < batch_size:
            raise ValueError("Generator must enforce seq_offsets[B+1] and L >= B")
        total_elements = total_length * k + batch_size * k * n + batch_size * n + total_length * n
        if total_elements > _MAX_TOTAL_ELEMENTS:
            raise ValueError("Generator must enforce the bmm total-element memory limit")

        device = _get_device()
        self._device = device

        self._jagged = jagged_input.to(device).contiguous()
        self._dense = dense_input.to(device).contiguous()
        self._bias = bias_input.to(device).contiguous()
        self._seq_offsets, self._max_seq_len = _build_offsets(batch_size, total_length, device)
        self._out = torch.empty((total_length, n), dtype=self._jagged.dtype, device=device)

        self._k = k
        self._n = n
        self._grid = lambda meta: (
            triton.cdiv(n, meta["BLOCK_N"]),
            triton.cdiv(self._max_seq_len, meta["BLOCK_M"]),
            batch_size,
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。autotune 自动注入 BLOCK_M / BLOCK_N / BLOCK_K。"""
        _synchronize(self._device)

        jagged_dense_bmm_broadcast_add_kernel[self._grid](
            seq_offsets=self._seq_offsets,
            Jagged=self._jagged,
            Dense=self._dense,
            Bias=self._bias,
            Out=self._out,
            AUTOTUNE_MAX_SEQ_LEN=self._max_seq_len,
            N=self._n,
            K=self._k,
            stride_jm=self._jagged.stride(0),
            stride_db=self._dense.stride(0),
            stride_dk=self._dense.stride(1),
            stride_dn=self._dense.stride(2),
            stride_bias_b=self._bias.stride(0),
            stride_om=self._out.stride(0),
            HAS_BIAS=True,
            ALLOW_TF32=False,
        )

        _synchronize(self._device)

        return self._out
