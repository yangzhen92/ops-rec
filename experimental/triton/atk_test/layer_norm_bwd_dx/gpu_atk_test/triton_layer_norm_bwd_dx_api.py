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

# triton_layer_norm_bwd_dx_api.py
"""
_layer_norm_bwd_dx kernel 的 ATK API 封装。

该 kernel 是 layer norm（非 learnable 分支）的反向，计算输入 X 的梯度 DX。
每个 program 处理一行：grid = (N,)。

Kernel 签名见 triton_layer_norm_bwd_dx_kernel.py。
  _layer_norm_bwd_dx(DX, DY, X, Mean, Rstd,
                     stride_dx, stride_dy, stride_x, D, eps,
                     BLOCK_D: tl.constexpr)

公式（对每一行 row）：
  xhat = (X[row] - Mean[row]) * Rstd[row]
  c1   = sum(xhat * DY[row]) / D
  c2   = sum(DY[row]) / D
  DX[row] = (DY[row] - (xhat * c1 + c2)) * Rstd[row]

Mean/Rstd 由 API 从 X 确定性推导（等价于前向 _layer_norm_fwd 的
COMPUTE_MEAN_AND_RSTD=True 分支），保证 GPU/NPU 两端输入一致。
"""

from __future__ import annotations


import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_layer_norm_bwd_dx_kernel import _layer_norm_bwd_dx

# pylint: disable=attribute-defined-outside-init

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
"""kernel 支持的 dtype。"""


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_layer_norm_bwd_dx requires an NPU or CUDA device")


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


@register("triton_layer_norm_bwd_dx")
@register("_layer_norm_bwd_dx")
class TritonLayerNormBwdDxApi(BaseApi):
    """_layer_norm_bwd_dx kernel 的 API 封装。

    只测非 learnable 分支：X/DY 2D 同形，Mean/Rstd 由 API 从 X 推导。
    """

    def init_by_input_data(self, input_data: InputDataset):
        # ── 1. 读取输入 ──
        x_input = input_data.kwargs.get("X")
        dy_input = input_data.kwargs.get("DY")
        eps = float(input_data.kwargs.get("eps", 1e-5))

        if not all(isinstance(v, torch.Tensor) for v in (x_input, dy_input)):
            raise ValueError("X and DY must both be tensors")

        x = x_input
        dy = dy_input

        if x.ndim != 2 or dy.ndim != 2:
            raise ValueError(f"X/DY must be 2D, got X {x.ndim}D, DY {dy.ndim}D")
        if x.shape != dy.shape:
            raise ValueError(f"X and DY must share shape, got X {x.shape}, DY {dy.shape}")

        n, d = x.shape
        if n == 0 or d == 0:
            raise ValueError(f"X/DY cannot have zero dim, got shape {x.shape}")

        # ── 2. 统一 dtype（以 X 为准）──
        if x.dtype == torch.float64:
            x = x.to(torch.float32).contiguous()
        if x.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported X dtype: {x.dtype}")
        dtype = x.dtype

        # 在 CPU 上完成对齐，避免 NPU Cast 限制；保证两端输入一致。
        x_cpu = x.cpu().to(dtype).contiguous()
        dy_cpu = dy.cpu().to(dtype).contiguous()

        # ── 3. 确定性推导 Mean/Rstd（等价前向计算）──
        # fp32 用 fp64 累加提高精度；fp16/bf16 用 fp32。结果统一转 fp32
        # （kernel 把 Mean/Rstd 读到 fp32 参与运算）。
        acc_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        xr = x_cpu.to(acc_dtype)
        mean_cpu = xr.mean(dim=1)  # [N]
        var_cpu = xr.var(dim=1, unbiased=False)  # [N]
        rstd_cpu = 1.0 / torch.sqrt(var_cpu + eps)  # [N]
        mean_cpu = mean_cpu.float().contiguous()
        rstd_cpu = rstd_cpu.float().contiguous()

        # ── 4. BLOCK_D / num_warps（对齐上游 triton_weighted_layer_norm_fwd）──
        # BLOCK_D = min(65536//elem_size, next_pow2(D))，必须覆盖整个 D。
        # 无硬编码上限：D 覆盖到 8192（YAML dim_values 上限）时 fp32 下
        # D 不超过 8192 时，fp32 的容量上限为 16384，分块大小由 D 的下一个二次幂决定。
        elem_size = x_cpu.element_size()
        max_fused_size = 65536 // elem_size
        block_d = min(max_fused_size, _next_pow2(d))
        if block_d < d:
            # 与上游一致：特征维过大（>= 64KB）时不支持，抛错而非静默截断。
            raise RuntimeError(f"This layer norm doesn't support feature dim D={d} (BLOCK_D would be {block_d} < D).")
        num_warps = min(max(block_d // 256, 1), 8)

        # ── 5. 移到目标设备 ──
        device = _get_device()
        x_d = x_cpu.to(device)
        dy_d = dy_cpu.to(device)
        mean_d = mean_cpu.to(device)
        rstd_d = rstd_cpu.to(device)
        dx = torch.empty_like(x_d, device=device)

        # 保存实例变量
        self._device = device
        self._n = n
        self._d = d
        self._eps = eps
        self._block_d = block_d
        self._num_warps = num_warps
        self._x = x_d
        self._dy = dy_d
        self._mean = mean_d
        self._rstd = rstd_d
        self._dx = dx
        self._stride_dx = dx.stride(0)
        self._stride_dy = dy_d.stride(0)
        self._stride_x = x_d.stride(0)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _synchronize(self._device)

        _layer_norm_bwd_dx[(self._n,)](
            self._dx,
            self._dy,
            self._x,
            self._mean,
            self._rstd,
            self._stride_dx,
            self._stride_dy,
            self._stride_x,
            self._d,
            self._eps,
            BLOCK_D=self._block_d,
            num_warps=self._num_warps,
        )

        _synchronize(self._device)

        return self._dx
