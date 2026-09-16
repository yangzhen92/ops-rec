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
# triton_split_2D_jagged_w_prefix_api.py
"""split_2D_jagged_w_prefix kernel 的 ATK 测试 API 封装。"""

from __future__ import annotations


import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_split_2D_jagged_w_prefix_kernel import split_2D_jagged_w_prefix

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MAX_TOTAL_ELEMENTS = 1 << 27
_MAX_CORE_DIM = 65535


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("split_2D_jagged_w_prefix requires an available NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _build_offsets(batch_size: int, total_length: int, device: str) -> torch.Tensor:
    """与精度 API 一致的确定性 offsets 生成。"""
    if batch_size < 1:
        raise ValueError(f"B must be >= 1, got {batch_size}")
    if total_length < 0:
        raise ValueError(f"L must be >= 0, got {total_length}")

    force_empty_segment = batch_size > 1 and (total_length + batch_size) % 5 == 0
    active_segments = batch_size - 1 if force_empty_segment else batch_size
    base_length, remainder = divmod(total_length, active_segments)
    empty_segment = (total_length + batch_size) % batch_size if force_empty_segment else -1

    offsets = [0]
    active_index = 0
    for batch_index in range(batch_size):
        if batch_index == empty_segment:
            segment_length = 0
        else:
            segment_length = base_length + int(active_index < remainder)
            active_index += 1
        offsets.append(offsets[-1] + segment_length)

    assert len(offsets) == batch_size + 1
    assert offsets[0] == 0
    assert offsets[-1] == total_length
    assert all(right >= left for left, right in zip(offsets, offsets[1:]))
    return torch.tensor(offsets, device=device, dtype=torch.int64)


@register("triton_split_2D_jagged_w_prefix")
class TritonSplit2DJaggedWPrefixPerfApi(BaseApi):
    """性能测试 API：使用原始 kernel 的单次 launch。"""

    def init_by_input_data(self, input_data: InputDataset):
        jagged_input = input_data.kwargs.get("JaggedIn")
        offsets_a_input = input_data.kwargs.get("OffsetsA")
        offsets_b_input = input_data.kwargs.get("OffsetsB")

        if not all(isinstance(v, torch.Tensor) for v in (jagged_input, offsets_a_input, offsets_b_input)):
            raise ValueError("JaggedIn, OffsetsA and OffsetsB must all be tensors")
        if jagged_input.ndim != 2:
            raise ValueError(f"JaggedIn must have shape [L, D], got {tuple(jagged_input.shape)}")
        if jagged_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported JaggedIn dtype: {jagged_input.dtype}")
        if offsets_a_input.ndim != 1 or offsets_a_input.numel() < 2 or offsets_a_input.dtype != torch.int64:
            raise ValueError("OffsetsA must be an int64 tensor with shape [B + 1]")
        if offsets_b_input.ndim != 1 or offsets_b_input.numel() < 2 or offsets_b_input.dtype != torch.int64:
            raise ValueError("OffsetsB must be an int64 tensor with shape [B + 1]")
        if offsets_a_input.numel() != offsets_b_input.numel():
            raise ValueError("OffsetsA and OffsetsB must have same length (same B)")

        total_length, d = jagged_input.shape
        batch_size = offsets_a_input.numel() - 1

        if d < 1:
            raise ValueError("D must be >= 1")
        if total_length < batch_size:
            raise ValueError("L must be >= B (each batch needs at least one token)")

        # 与精度 API 一致的 la/lb 划分逻辑
        la = max(batch_size, total_length // 2)
        lb = total_length - la
        if lb < batch_size:
            lb = batch_size
            la = total_length - lb
        if la < batch_size:
            la = batch_size
            lb = total_length - la
        if la + lb != total_length:
            lb = total_length - la
            if lb < batch_size:
                la = total_length - batch_size
                lb = batch_size

        device = _get_device()
        self._jagged = jagged_input.to(device).contiguous()
        # 用确定性算法重建 offsets（JSON 中的 OffsetsA/OffsetsB 数值是随机生成的无意义值）
        self._offsets_a = _build_offsets(batch_size, la, device)
        self._offsets_b = _build_offsets(batch_size, lb, device)

        max_seq_len_a = int((self._offsets_a[1:] - self._offsets_a[:-1]).max().item())
        max_seq_len_b = int((self._offsets_b[1:] - self._offsets_b[:-1]).max().item())
        self._max_seq_len = max_seq_len_a + max_seq_len_b

        self._out_a = torch.empty((la, d), device=device, dtype=self._jagged.dtype)
        self._out_b = torch.empty((lb, d), device=device, dtype=self._jagged.dtype)

        core_dim = self._max_seq_len * batch_size
        if core_dim > _MAX_CORE_DIM:
            raise ValueError(
                f"coreDim={core_dim} exceeds NPU limit {_MAX_CORE_DIM} "
                f"(B={batch_size}, max_seq_len={self._max_seq_len})"
            )

        self._grid = (self._max_seq_len, batch_size)
        self._d = d
        self._device = device
        self._batch_size = batch_size
        self._la = la
        self._lb = lb
        self._n_prefix_to_B = 0

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)
        split_2D_jagged_w_prefix[self._grid](
            JaggedIn=self._jagged,
            DenseSize=0,
            OffsetsA=self._offsets_a,
            OffsetsB=self._offsets_b,
            OutA=self._out_a,
            OutB=self._out_b,
            D=self._d,
            stride_id=self._jagged.stride(0),
            stride_ad=self._out_a.stride(0),
            stride_bd=self._out_b.stride(0),
            n_prefix_to_B=self._n_prefix_to_B,
            IS_DENSE_A=False,
            IS_DENSE_B=False,
            IS_REPLACE=False,
        )
        _synchronize(self._device)
        return self._out_a, self._out_b
