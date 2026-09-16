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
# triton_concat_2D_jagged_w_prefix_api.py

"""concat_2D_jagged_w_prefix 的 ATK API 封装（autotune 版本）。

直接调用 concat_2D_jagged_w_prefix，按 batch 将 ValuesA、ValuesB 拼接：
  grid = (max_seq_len, B)，其中 max_seq_len = max_a + max_b
  输出 Out [L_A + L_B, D]

使用要点：
  1. triton_jagged.py 中的 concat_2D_jagged_w_prefix 带 @triton.autotune
     （BLOCK_D 16/32/64 候选），D 维由 BLOCK_D 循环分块、各 chunk 有
     offs_d < D 掩码保护；BLOCK_D 由 autotune 自动注入，无需手动指定。
  2. 本 API 覆盖 IS_DENSE_A=False、IS_DENSE_B=False、IS_REPLACE=False、
     n_prefix_from_B=0 的 jagged+jagged 拼接路径。
"""

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_concat_2D_jagged_w_prefix_kernel import concat_2D_jagged_w_prefix

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MAX_TOTAL_ELEMENTS = 1 << 27
_MAX_CORE_DIM = 65535


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("concat_2D_jagged_w_prefix requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _build_both_jagged_offsets(
    batch_size: int, total_length_a: int, total_length_b: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """生成两套严格递增的 offsets，确保每一段至少一个 token。

    返回 (offsets_a, offsets_b, max_seq_len_a, max_seq_len_b)。
    对于 IS_REPLACE=False 的拼接模式，grid 需要 max_seq_len = max_a + max_b
    来覆盖所有输出 token。
    """
    if batch_size < 1:
        raise ValueError(f"B must be >= 1, got {batch_size}")
    if total_length_a < batch_size:
        raise ValueError(f"L_A must be >= B, got L_A={total_length_a}, B={batch_size}")
    if total_length_b < batch_size:
        raise ValueError(f"L_B must be >= B, got L_B={total_length_b}, B={batch_size}")

    def _build(length: int, seed_offset: int) -> tuple[list[int], int]:
        remaining = length - batch_size
        weights = [1 + ((i * 7 + seed_offset) % 11) for i in range(batch_size)]
        weight_sum = sum(weights)
        extra = [remaining * w // weight_sum for w in weights]
        residue = remaining - sum(extra)
        for ri in range(residue):
            extra[(ri * 5 + seed_offset) % batch_size] += 1

        offsets = [0]
        max_len = 0
        for bi in range(batch_size):
            seg = 1 + extra[bi]
            offsets.append(offsets[-1] + seg)
            max_len = max(max_len, seg)
        return offsets, max_len

    offsets_a_list, max_a = _build(total_length_a, 0)
    offsets_b_list, max_b = _build(total_length_b, 3)

    assert offsets_a_list[0] == 0 and offsets_a_list[-1] == total_length_a
    assert offsets_b_list[0] == 0 and offsets_b_list[-1] == total_length_b
    assert all(right > left for left, right in zip(offsets_a_list, offsets_a_list[1:]))

    return (
        torch.tensor(offsets_a_list, dtype=torch.int64, device=device),
        torch.tensor(offsets_b_list, dtype=torch.int64, device=device),
        max_a,
        max_b,
    )


@register("triton_concat_2D_jagged_w_prefix")
class TritonConcat2DJaggedWPrefixApi(BaseApi):
    """concat_2D_jagged_w_prefix 的 Autotune API 封装。"""

    def init_by_input_data(self, input_data: InputDataset):
        """根据 ATK InputDataset 组装 kernel 入参（不计时）。"""
        offsets_input = input_data.kwargs.get("seq_offsets")
        values_a_input = input_data.kwargs.get("ValuesA")
        values_b_input = input_data.kwargs.get("ValuesB")

        if not all(isinstance(v, torch.Tensor) for v in (offsets_input, values_a_input, values_b_input)):
            raise ValueError("seq_offsets, ValuesA and ValuesB must all be tensors")
        if offsets_input.ndim != 1 or offsets_input.numel() < 2 or offsets_input.dtype != torch.int64:
            raise ValueError("seq_offsets must be an int64 tensor with shape [B + 1]")
        if values_a_input.ndim != 2 or values_b_input.ndim != 2:
            raise ValueError("ValuesA and ValuesB must be 2D tensors [L, D]")
        if values_a_input.shape[1] != values_b_input.shape[1]:
            raise ValueError("ValuesA and ValuesB must have same D dimension")
        if values_a_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {values_a_input.dtype}")
        if values_b_input.dtype != values_a_input.dtype:
            raise ValueError("ValuesA and ValuesB must share the same dtype")

        la, d = values_a_input.shape
        lb, _ = values_b_input.shape
        batch_size = offsets_input.numel() - 1

        if d < 1:
            raise ValueError("D must be >= 1")
        if la < batch_size or lb < batch_size:
            raise ValueError("Both L_A and L_B must be >= batch_size to avoid empty batches")
        if (la + lb) * d * 2 > _MAX_TOTAL_ELEMENTS:
            raise ValueError("Total elements exceed limit")

        device = _get_device()
        self._values_a = values_a_input.to(device).contiguous()
        self._values_b = values_b_input.to(device).contiguous()

        # 由 shape 重建 offsets，确保 GPU/NPU 两端一致
        self._offsets_a, self._offsets_b, max_a, max_b = _build_both_jagged_offsets(batch_size, la, lb, device)

        # IS_REPLACE=False（拼接模式）下输出序列长度 = seq_len_a + seq_len_b，
        # 因此 grid 的 max_seq_len 必须是每 batch 最大总长度（max_a + max_b）。
        self._max_seq_len = max_a + max_b

        # Out shape: [L_A + L_B, D]
        self._out = torch.empty((la + lb, d), device=device, dtype=self._values_a.dtype)

        # coreDim = max_seq_len * B (二维 grid)
        core_dim = self._max_seq_len * batch_size
        if core_dim > _MAX_CORE_DIM:
            raise ValueError(
                f"coreDim={core_dim} exceeds NPU limit {_MAX_CORE_DIM} "
                f"(B={batch_size}, max_seq_len={self._max_seq_len})"
            )

        self._d = d
        self._device = device
        self._batch_size = batch_size
        self._grid = (self._max_seq_len, batch_size)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。autotune 自动注入 BLOCK_D。"""
        _synchronize(self._device)

        concat_2D_jagged_w_prefix[self._grid](
            OffsetsA=self._offsets_a,
            ValuesA=self._values_a,
            OffsetsB=self._offsets_b,
            ValuesB=self._values_b,
            DenseSize=0,
            Out=self._out,
            D=self._d,
            stride_ad=self._values_a.stride(0),
            stride_bd=self._values_b.stride(0),
            stride_dense_batch=0,
            stride_od=self._out.stride(0),
            n_prefix_from_B=0,
            IS_DENSE_A=False,
            IS_DENSE_B=False,
            BLOCK_D=triton.next_power_of_2(self._d),
            IS_REPLACE=False,
        )

        _synchronize(self._device)
        return self._out
