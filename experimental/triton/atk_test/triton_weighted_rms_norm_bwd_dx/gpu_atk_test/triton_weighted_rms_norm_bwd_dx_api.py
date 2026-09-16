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
# triton_weighted_rms_norm_bwd_dx_api.py
"""
_weighted_rms_norm_bwd_dx kernel ATK API 封装。

直接调用 _weighted_rms_norm_bwd_dx，对 RMSNorm 反向传播的输入梯度 DY 求 DX：
  grid = (N,)，每个 program 处理一行
    xhat = x * rstd
    wdy = w * dy
    c1 = sum(xhat * wdy) / D
    dx = (wdy - xhat * c1) * rstd
  并用 atomics + lock 把 partial weight grad (dy * xhat) 累加到 DW [GROUP_N, D]。
"""

from __future__ import annotations


import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_weighted_rms_norm_bwd_dx_kernel import _weighted_rms_norm_bwd_dx

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
"""kernel 支持并参与精度对比的输入 dtype。"""


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_weighted_rms_norm_bwd_dx requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


@register("triton_weighted_rms_norm_bwd_dx")
class WeightedRmsNormBwdDxApi(BaseApi):
    """_weighted_rms_norm_bwd_dx kernel 的 API 封装。"""

    def init_by_input_data(self, input_data: InputDataset):
        """根据 ATK InputDataset 组装 kernel 入参（不计时）。"""
        # ── 1. 读取输入参数 ──
        x_input = input_data.kwargs.get("X")
        dy_input = input_data.kwargs.get("DY")
        w_input = input_data.kwargs.get("W")
        rstd_input = input_data.kwargs.get("Rstd")
        eps_input = input_data.kwargs.get("eps")

        # ── 2. 输入校验 ──
        if not all(isinstance(v, torch.Tensor) for v in (x_input, dy_input, w_input, rstd_input)):
            raise ValueError("X, DY, W, Rstd must all be tensors")
        if x_input.ndim != 2 or dy_input.ndim != 2:
            raise ValueError(f"Expected X, DY to be 2D, got X {x_input.ndim}D, DY {dy_input.ndim}D")
        if w_input.ndim != 1:
            raise ValueError(f"Expected W to be 1D, got {w_input.ndim}D")
        if x_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {x_input.dtype}")
        if x_input.shape[0] == 0:
            raise ValueError("X must have non-zero row count")

        n, d = x_input.shape
        if dy_input.shape != x_input.shape:
            raise ValueError(f"DY shape {dy_input.shape} != X shape {x_input.shape}")
        if w_input.shape[0] != d:
            raise ValueError(f"W first dim {w_input.shape[0]} != X second dim {d}")

        # ── 3. 获取设备 ──
        device = _get_device()
        self._device = device

        # ── 4. 搬数据到设备，dtype 归一 ──
        if x_input.dtype == torch.float64:
            x = x_input.to(torch.float32).contiguous()
        else:
            x = x_input.contiguous()
        x = x.to(device)
        dtype = x.dtype
        element_size = x.element_size()

        dy = dy_input.to(device).contiguous()
        if dy.dtype != dtype:
            dy = dy.to(dtype)

        w = w_input.to(device).contiguous()
        if w.dtype != dtype:
            w = w.to(dtype)

        rstd = rstd_input.to(device).contiguous()
        if rstd.dtype != torch.float32:
            rstd = rstd.to(torch.float32)

        # ── 5. 计算 kernel 参数 ──
        # BLOCK_D / num_warps（与原生 wrapper 一致）
        max_fused_size = 65536 // element_size
        block_d = min(max_fused_size, triton.next_power_of_2(d))
        num_warps = min(max(block_d // 256, 1), 8)

        # GROUP_N（与原生 wrapper backward 中的计算一致）
        if d <= 1024:
            group_n = 256 * 8
        elif d <= 4096:
            group_n = 128 * 8
        elif d <= 8192:
            group_n = 96 * 8
        else:
            group_n = 64 * 8
        group_n = n if group_n > n else group_n

        # eps
        eps = float(eps_input) if eps_input is not None else 1e-5

        # ── 6. 分配内部 buffer ──
        dx = torch.empty((n, d), dtype=dtype, device=device).contiguous()
        dw = torch.empty((group_n, d), dtype=torch.float32, device=device)
        locks = torch.zeros(2 * group_n, dtype=torch.int32, device=device)

        # grid = (N,)
        grid = (n,)

        self._x = x
        self._dy = dy
        self._w = w
        self._rstd = rstd
        self._dx = dx
        self._dw = dw
        self._locks = locks
        self._d = d
        self._group_n = group_n
        self._eps = eps
        self._stride_dx = dx.stride(0)
        self._stride_dy = dy.stride(0)
        self._stride_x = x.stride(0)
        self._block_d = block_d
        self._num_warps = num_warps
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。"""
        _synchronize(self._device)

        _weighted_rms_norm_bwd_dx[self._grid](
            self._dx,
            self._dy,
            self._dw,
            self._x,
            self._w,
            self._rstd,
            self._locks,
            self._stride_dx,
            self._stride_dy,
            self._stride_x,
            self._d,
            self._eps,
            GROUP_N=self._group_n,
            BLOCK_D=self._block_d,
            num_warps=self._num_warps,
        )

        _synchronize(self._device)

        return self._dx
