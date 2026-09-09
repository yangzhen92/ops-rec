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

"""
_layer_norm_bwd_dx kernel ATK API 封装 — Autotune 版本。

直接调用 _layer_norm_bwd_dx，对输入 X [N, D] 的每一行计算输入梯度 DX：
  grid = (N,)，每个 program 处理一行

公式（对每一行 row）：
  xhat = (X[row] - Mean[row]) * Rstd[row]
  c1   = sum(xhat * DY[row]) / D
  c2   = sum(DY[row]) / D
  DX[row] = (DY[row] - (xhat * c1 + c2)) * Rstd[row]

与 double_benchmark 版 API 的区别：
  1. 导入的源文件（autotune/triton_layer_norm.py）中的 _layer_norm_bwd_dx
     带 @triton.autotune（BLOCK_D / num_warps 候选 + early_config_prune 按 D 剪枝）。
  2. 不再主动计算 / 传递 BLOCK_D 与 num_warps，由 autotune 自动注入。
"""

from __future__ import annotations


import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_layer_norm import _layer_norm_bwd_dx

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


@register("triton_layer_norm_bwd_dx")
# @register("triton_layer_norm_bwd_dx.autotune")
# @register("triton_layer_norm._layer_norm_bwd_dx.autotune")
# @register("_layer_norm_bwd_dx_autotune")
class TritonLayerNormBwdDxAutotuneApi(BaseApi):
    """_layer_norm_bwd_dx kernel 的 Autotune API 封装。

    只测非 learnable 分支：X/DY 2D 同形，Mean/Rstd 由 API 从 X 推导。
    BLOCK_D / num_warps 由 @triton.autotune 自动注入，不在此计算/传递。
    """

    def __init__(self):
        self._device = None
        self._n = 0
        self._d = 0
        self._eps = 0.0
        self._x = None
        self._dy = None
        self._mean = None
        self._rstd = None
        self._dx = None
        self._stride_dx = 0
        self._stride_dy = 0
        self._stride_x = 0

    def init_by_input_data(self, input_data: InputDataset):
        # ── 1. 读取输入 ──
        x_input = input_data.kwargs.get("X")
        dy_input = input_data.kwargs.get("DY")
        eps = float(input_data.kwargs.get("eps", 1e-5))

        if not all(isinstance(v, torch.Tensor) for v in (x_input, dy_input)):
            raise ValueError("X and DY must both be tensors")

        if x_input.ndim != 2 or dy_input.ndim != 2:
            raise ValueError(f"X/DY must be 2D, got X {x_input.ndim}D, DY {dy_input.ndim}D")
        if x_input.shape != dy_input.shape:
            raise ValueError(f"X and DY must share shape, got X {x_input.shape}, DY {dy_input.shape}")

        n, d = x_input.shape
        if n == 0 or d == 0:
            raise ValueError(f"X/DY cannot have zero dim, got shape {x_input.shape}")

        # ── 2. 统一 dtype（以 X 为准）──
        # double_benchmark 模式下 ATK 可能生成 fp64 输入；kernel 只支持
        # fp32/fp16/bf16，且 NPU 不支持 fp64 Cast。需在 CPU 上先降到 fp32。
        if x_input.dtype == torch.float64:
            x_input = x_input.to(torch.float32).contiguous()
        if x_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported X dtype: {x_input.dtype}")
        dtype = x_input.dtype

        # 在 CPU 上完成对齐，避免 NPU Cast 限制；保证两端输入一致。
        x_cpu = x_input.cpu().to(dtype).contiguous()
        dy_cpu = dy_input.cpu().to(dtype).contiguous()

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

        # ── 4. 移到目标设备 ──
        device = _get_device()
        x_d = x_cpu.to(device)
        dy_d = dy_cpu.to(device)
        mean_d = mean_cpu.to(device)
        rstd_d = rstd_cpu.to(device)
        dx = torch.empty_like(x_d, device=device)

        # 保存实例变量（注意：不保存 BLOCK_D / num_warps，autotune 会按 D 自动选择）
        self._device = device
        self._n = n
        self._d = d
        self._eps = eps
        self._x = x_d
        self._dy = dy_d
        self._mean = mean_d
        self._rstd = rstd_d
        self._dx = dx
        self._stride_dx = dx.stride(0)
        self._stride_dy = dy_d.stride(0)
        self._stride_x = x_d.stride(0)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。autotune 自动注入 BLOCK_D / num_warps。"""
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
            # BLOCK_D / num_warps 由 autotune 注入，不显式传递
        )

        _synchronize(self._device)

        return self._dx
