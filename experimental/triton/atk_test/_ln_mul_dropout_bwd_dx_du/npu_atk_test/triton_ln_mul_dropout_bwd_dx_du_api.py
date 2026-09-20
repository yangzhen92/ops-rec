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

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_ln_mul_dropout_bwd_dx_du_kernel import _ln_mul_dropout_bwd_dx_du

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_ln_mul_dropout_bwd_dx_du requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _compute_tile_num(n: int, device: str) -> int:
    if device == "npu":
        properties = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        compute_units = properties.get("num_vectorcore")
    else:
        compute_units = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if not isinstance(compute_units, int) or compute_units <= 0:
        raise RuntimeError(f"Unable to determine compute unit count for {device}")
    return max(1, min(compute_units * 64, n // 4))


@register("triton_ln_mul_dropout_bwd_dx_du")
class LnMulDropoutBwdDxDuAutotuneApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x_input = input_data.kwargs.get("X")
        u_input = input_data.kwargs.get("U")
        dy_input = input_data.kwargs.get("DY")
        w_input = input_data.kwargs.get("W")
        b_input = input_data.kwargs.get("B")
        mean_input = input_data.kwargs.get("Mean")
        rstd_input = input_data.kwargs.get("Rstd")

        device = _get_device()

        eps = float(input_data.kwargs.get("eps", 1e-5))
        dropout_ratio = float(input_data.kwargs.get("dropout_ratio", 0.0))
        training = bool(input_data.kwargs.get("TRAINING", False))
        concat_ux = bool(input_data.kwargs.get("CONCAT_UX", False))
        compute_y = False
        seed = input_data.kwargs.get("seed", None)
        if seed is None:
            seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()

        if not all(isinstance(v, torch.Tensor) for v in (x_input, u_input, dy_input)):
            raise ValueError("X, U, DY must all be tensors")

        if x_input.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got {x_input.ndim}D")
        if u_input.ndim != 2:
            raise ValueError(f"Expected U to be 2D, got {u_input.ndim}D")
        if dy_input.ndim != 2:
            raise ValueError(f"Expected DY to be 2D, got {dy_input.ndim}D")
        if x_input.shape != u_input.shape:
            raise ValueError(f"X shape {x_input.shape} must match U shape {u_input.shape}")

        n, d = x_input.shape
        if d > 4096:
            raise ValueError(f"Expected feature dimension D <= 4096, got {d}")
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError(f"dropout_ratio must be in [0, 1), got {dropout_ratio}")

        if dy_input.shape[0] != n:
            raise ValueError(f"DY rows {dy_input.shape[0]} != X rows {n}")
        if concat_ux:
            if dy_input.shape[1] != 3 * d:
                raise ValueError(f"CONCAT_UX=True: expected DY dim={3 * d}, got {dy_input.shape[1]}")
        else:
            if dy_input.shape[1] != d:
                raise ValueError(f"DY dim {dy_input.shape[1]} != X dim {d}")

        if w_input is not None:
            w = w_input
            if w.dim() != 1 or w.numel() != d:
                raise ValueError(f"W must be 1D with {d} elements")
        else:
            w = torch.ones(d, dtype=x_input.dtype, device=device)

        if b_input is not None:
            b = b_input
            if b.dim() != 1 or b.numel() != d:
                raise ValueError(f"B must be 1D with {d} elements")
        else:
            b = torch.zeros(d, dtype=x_input.dtype, device=device)

        x_cpu = x_input.detach().cpu().float()

        if mean_input is not None:
            mean = mean_input
            if mean.numel() != n:
                raise ValueError(f"Mean numel {mean.numel()} != N {n}")
        else:
            mean = x_cpu.mean(dim=1, keepdim=False)

        if rstd_input is not None:
            rstd = rstd_input
            if rstd.numel() != n:
                raise ValueError(f"Rstd numel {rstd.numel()} != N {n}")
        else:
            var = x_cpu.var(dim=1, unbiased=False, keepdim=False)
            rstd = 1.0 / torch.sqrt(var + eps)

        num_u_slices = 1
        ud = d

        x = x_input.to(torch.float32).contiguous() if x_input.dtype == torch.float64 else x_input.contiguous()
        u = u_input.to(torch.float32).contiguous() if u_input.dtype == torch.float64 else u_input.contiguous()
        dy = dy_input.to(torch.float32).contiguous() if dy_input.dtype == torch.float64 else dy_input.contiguous()
        w = w.to(torch.float32).contiguous() if w.dtype == torch.float64 else w.contiguous()
        b = b.to(torch.float32).contiguous() if b.dtype == torch.float64 else b.contiguous()
        x = x.to(device)
        u = u.to(device)
        dy = dy.to(device)
        w = w.to(device)
        b = b.to(device)
        mean = mean.to(torch.float32).contiguous().to(device)
        rstd = rstd.to(torch.float32).contiguous().to(device)

        dx = torch.empty_like(x, device=device)
        du = torch.empty_like(u, device=device)

        tile_num = _compute_tile_num(n, device)
        _dweight = torch.empty((tile_num, d), dtype=torch.float32, device=device)
        _dbias = torch.empty((tile_num, d), dtype=torch.float32, device=device)

        y = None
        if concat_ux:
            y = torch.empty((n, 3 * d), dtype=x.dtype, device=device)
        else:
            y = torch.empty_like(x, device=device)

        stride_dx = dx.stride(0)
        stride_du0 = du.stride(0)
        stride_du1 = 1
        stride_dy = dy.stride(0)
        stride_x = x.stride(0)
        stride_u0 = u.stride(0)
        stride_u1 = 1
        stride_y = y.stride(0) if y is not None else 0

        self._device = device
        self._dx = dx
        self._du = du
        self._dy = dy
        self._dweight = _dweight
        self._dbias = _dbias
        self._x = x
        self._u = u
        self._y = y
        self._w = w
        self._b = b
        self._mean = mean
        self._rstd = rstd
        self._d = d
        self._eps = eps
        self._seed = seed
        self._dropout_ratio = dropout_ratio
        self._n = n
        self._num_u_slices = num_u_slices
        self._ud = ud
        self._tile_num = tile_num
        self._training = training
        self._concat_ux = concat_ux
        self._compute_y = compute_y
        self._stride_dx = stride_dx
        self._stride_du0 = stride_du0
        self._stride_du1 = stride_du1
        self._stride_dy = stride_dy
        self._stride_x = stride_x
        self._stride_u0 = stride_u0
        self._stride_u1 = stride_u1
        self._stride_y = stride_y

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        _ln_mul_dropout_bwd_dx_du[(self._tile_num,)](
            DX=self._dx,
            DU=self._du,
            DY=self._dy,
            DW=self._dweight,
            DB=self._dbias,
            X=self._x,
            U=self._u,
            Y=self._y,
            W=self._w,
            B=self._b,
            Mean=self._mean,
            Rstd=self._rstd,
            stride_dx=self._stride_dx,
            stride_du0=self._stride_du0,
            stride_du1=self._stride_du1,
            stride_dy=self._stride_dy,
            stride_x=self._stride_x,
            stride_u0=self._stride_u0,
            stride_u1=self._stride_u1,
            stride_y=self._stride_y,
            D=self._d,
            eps=self._eps,
            seed=self._seed,
            dropout_ratio=self._dropout_ratio,
            N=self._n,
            NUM_U_SLICES=self._num_u_slices,
            UD=self._ud,
            TRAINING=self._training,
            CONCAT_UX=self._concat_ux,
            COMPUTE_Y=self._compute_y,
        )

        _synchronize(self._device)

        return self._dx, self._du
