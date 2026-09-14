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
# triton_weighted_rms_norm_fwd_api.py
"""ATK API wrapper for the GPU weighted RMS normalization kernel."""

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_weighted_rms_norm_fwd_kernel import _weighted_rms_norm_fwd


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_weighted_rms_norm_fwd requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


@register("triton_weighted_rms_norm_fwd")
@register("_weighted_rms_norm_fwd")
class WeightedRmsNormFwdApi(BaseApi):
    """_weighted_rms_norm_fwd kernel 的 API 封装。"""

    # pylint: disable=attribute-defined-outside-init

    def init_by_input_data(self, input_data: InputDataset):
        """Prepare kernel arguments from an ATK input dataset."""
        x_input = input_data.kwargs.get("X")
        w_input = input_data.kwargs.get("W")
        y_input = input_data.kwargs.get("Y")
        rstd_input = input_data.kwargs.get("Rstd")
        eps_input = input_data.kwargs.get("eps")
        if not all(isinstance(v, torch.Tensor) for v in (x_input, w_input, y_input, rstd_input)):
            raise ValueError("X, W, Y, Rstd must all be tensors")
        if x_input.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got {x_input.ndim}D")
        if w_input.ndim != 1:
            raise ValueError(f"Expected W to be 1D, got {w_input.ndim}D")
        if x_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {x_input.dtype}")
        if x_input.shape[0] == 0:
            raise ValueError("X must have non-zero row count")

        n, d = x_input.shape
        if w_input.shape[0] != d:
            raise ValueError(f"W first dim {w_input.shape[0]} != X second dim {d}")

        device = _get_device()
        self._device = device

        x = x_input.contiguous()
        x = x.to(device)
        dtype = x.dtype
        element_size = x.element_size()

        w = w_input.to(device).contiguous()
        if w.dtype != dtype:
            w = w.to(dtype)

        max_fused_size = 65536 // element_size
        block_d = min(max_fused_size, triton.next_power_of_2(d))
        num_warps = min(max(block_d // 256, 1), 8)

        y = torch.empty((n, d), dtype=dtype, device=device).contiguous()
        rstd = torch.empty((n,), dtype=torch.float32, device=device)
        grid = (n,)
        eps = float(eps_input) if eps_input is not None else 1e-5
        self._x = x
        self._w = w
        self._y = y
        self._rstd = rstd
        self._d = d
        self._eps = eps
        self._stride_x = x.stride(0)
        self._stride_y = y.stride(0)
        self._block_d = block_d
        self._num_warps = num_warps
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。"""
        _synchronize(self._device)

        _weighted_rms_norm_fwd[self._grid](
            self._x,
            self._y,
            self._w,
            self._rstd,
            self._d,
            self._eps,
            self._stride_x,
            self._stride_y,
            BLOCK_D=self._block_d,
            num_warps=self._num_warps,
        )

        _synchronize(self._device)
        return self._y, self._rstd
