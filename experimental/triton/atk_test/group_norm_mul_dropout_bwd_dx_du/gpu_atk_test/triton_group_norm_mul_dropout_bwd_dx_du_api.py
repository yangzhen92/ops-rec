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

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_group_norm_mul_dropout_bwd_dx_du_kernel import _group_norm_mul_dropout_bwd_dx_du

# pylint: disable=attribute-defined-outside-init


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


_MAX_FUSED_WORDS = 16384
_MAX_BLOCK_D = 4096


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_group_norm_mul_dropout_bwd_dx_du requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@register("triton_group_norm_mul_dropout_bwd_dx_du")
@register("_group_norm_mul_dropout_bwd_dx_du")
class TritonGroupNormMulDropoutBwdDxDuApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x_input = input_data.kwargs.get("X")
        u_input = input_data.kwargs.get("U")
        w_input = input_data.kwargs.get("W")
        b_input = input_data.kwargs.get("B")
        dy_input = input_data.kwargs.get("DY")
        eps = float(input_data.kwargs.get("eps", 1e-5))
        dropout_ratio = float(input_data.kwargs.get("dropout_ratio", 0.0))
        training = bool(float(input_data.kwargs.get("TRAINING", 0.0)))
        concat_ux = bool(float(input_data.kwargs.get("CONCAT_UX", 0.0)))
        seed = input_data.kwargs.get("seed", None)

        training = False
        concat_ux = False
        dropout_ratio = 0.0
        compute_y = False

        if not all(isinstance(v, torch.Tensor) for v in (x_input, u_input, w_input, b_input, dy_input)):
            raise ValueError("X, U, W, B, DY must all be tensors")

        x = x_input
        u = u_input
        w = w_input
        b = b_input
        dy = dy_input

        if x.ndim != 2 or u.ndim != 2 or dy.ndim != 2:
            raise ValueError(f"X/U/DY must be 2D, got X {x.ndim}D, U {u.ndim}D, DY {dy.ndim}D")
        if x.shape != u.shape or x.shape != dy.shape:
            raise ValueError(f"X/U/DY must share shape, got X {x.shape}, U {u.shape}, DY {dy.shape}")

        n, inner = x.shape

        _HEAD_CANDIDATES = [1, 2, 4, 8, 16, 32]
        num_heads = None
        for candidate in _HEAD_CANDIDATES:
            if inner % candidate == 0:
                num_heads = candidate
                break
        if num_heads is None:
            num_heads = 1
        linear_dim = inner // num_heads
        if linear_dim <= 0:
            raise ValueError(f"linear_dim must be positive, got {linear_dim}")

        device = _get_device()

        if dy.ndim != 2 or list(dy.shape) != [n, inner]:
            raise ValueError(f"DY shape {tuple(dy.shape)} must be [N, inner]=[{n}, {inner}]")

        def _dtype_map(t):
            if t.dtype == torch.float64:
                return torch.float32
            return t.dtype

        dtype = _dtype_map(x)
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported X dtype: {dtype}")

        def _aligned(t):
            t = t.to(dtype)
            return t.contiguous()

        x_d = _aligned(x)
        u_d = _aligned(u)
        dy_d = _aligned(dy)

        if isinstance(w, torch.Tensor) and w.numel() == num_heads:
            w_d = _aligned(w)
        else:
            w_d = torch.ones(num_heads, dtype=dtype)
        if isinstance(b, torch.Tensor) and b.numel() == num_heads:
            b_d = _aligned(b)
        else:
            b_d = torch.zeros(num_heads, dtype=dtype)

        x_cpu = x_d.cpu()
        if dtype == torch.float32:
            acc_dtype = torch.float64
        else:
            acc_dtype = torch.float32
        xr = x_cpu.to(acc_dtype).reshape(n, num_heads, linear_dim)
        mean_h = xr.mean(dim=2, keepdim=False)  # [N, H]
        var_h = xr.var(dim=2, unbiased=False, keepdim=False)  # [N, H]
        rstd_h = 1.0 / torch.sqrt(var_h + eps)

        mean = mean_h.reshape(n * num_heads).float().contiguous()
        rstd = rstd_h.reshape(n * num_heads).float().contiguous()

        if linear_dim > _MAX_BLOCK_D:
            raise ValueError(
                f"linear_dim {linear_dim} exceeds max supported BLOCK_D {_MAX_BLOCK_D}; "
                f"this case cannot be executed correctly and should be filtered by generator."
            )
        block_d = _next_pow2(linear_dim)
        block_h = _next_pow2(num_heads)
        if block_d * block_h > _MAX_FUSED_WORDS:
            while block_d > 1 and block_d * block_h > _MAX_FUSED_WORDS:
                block_d >>= 1

        if block_d < linear_dim:
            raise ValueError(f"BLOCK_D {block_d} < linear_dim {linear_dim}; cannot cover full feature dim")
        block_d = min(block_d, _MAX_BLOCK_D)
        num_warps = min(max(block_d * block_h // 256, 1), 8)

        dim = inner
        if dim <= 1024:
            group_n = 256 * 8
        elif dim <= 4096:
            group_n = 128 * 8
        elif dim <= 8192:
            group_n = 96 * 8
        else:
            group_n = 64 * 8
        group_n = n if group_n > n else group_n

        x_d = x_d.to(device)
        u_d = u_d.to(device)
        w_d = w_d.to(device)
        b_d = b_d.to(device)
        dy_d = dy_d.to(device)
        mean_d = mean.to(device)
        rstd_d = rstd.to(device)

        dx = torch.empty_like(x_d, device=device)
        du = torch.empty_like(u_d, device=device)
        _dweight = torch.zeros((group_n, num_heads), dtype=torch.float32, device=device)
        _dbias = torch.zeros((group_n, num_heads), dtype=torch.float32, device=device)
        y = None if not compute_y else torch.empty((n, inner), dtype=x_d.dtype, device=device)

        if seed is None:
            seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()

        self._device = device
        self._n = n
        self._linear_dim = linear_dim
        self._num_heads = num_heads
        self._block_d = block_d
        self._block_h = block_h
        self._num_warps = num_warps
        self._group_n = group_n
        self._eps = eps
        self._seed = seed
        self._dropout_ratio = dropout_ratio
        self._training = training
        self._concat_ux = concat_ux
        self._compute_y = compute_y
        self._dx = dx
        self._du = du
        self._dy = dy_d
        self._dweight = _dweight
        self._dbias = _dbias
        self._x = x_d
        self._u = u_d
        self._y = y
        self._w = w_d
        self._b = b_d
        self._mean = mean_d
        self._rstd = rstd_d
        self._grid = (n,)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        _group_norm_mul_dropout_bwd_dx_du[self._grid](
            self._dx,
            self._du,
            self._dy,
            self._dweight,
            self._dbias,
            self._x,
            self._u,
            self._y,
            self._w,
            self._b,
            self._mean,
            self._rstd,
            self._dx.stride(0),
            self._du.stride(0),
            self._dy.stride(0),
            self._x.stride(0),
            self._u.stride(0),
            self._y.stride(0) if self._compute_y else 0,
            self._linear_dim,
            self._num_heads,
            self._eps,
            self._seed,
            self._dropout_ratio,
            GROUP_N=self._group_n,
            BLOCK_D=self._block_d,
            BLOCK_H=self._block_h,
            TRAINING=self._training,
            CONCAT_UX=self._concat_ux,
            COMPUTE_Y=self._compute_y,
            num_warps=self._num_warps,
        )

        _synchronize(self._device)

        return self._dx, self._du
