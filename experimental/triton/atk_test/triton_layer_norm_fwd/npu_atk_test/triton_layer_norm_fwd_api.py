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
_layer_norm_fwd kernel ATK API 封装 — Autotune 版本。

直接调用 _layer_norm_fwd，对输入 X [N, D] 的每一行做 layer normalization：
  grid = (N,)，每个 program 处理一行
  输出 Y [N, D]（归一化结果），并产出 Mean [N] / Rstd [N]（fp32）

与 double_benchmark 版 API 的区别：
  1. 导入 triton_layer_norm_fwd_kernel.py 中的 _layer_norm_fwd
     带 @triton.autotune（BLOCK_D / num_warps 候选 + early_config_prune 按 D 剪枝）。
  2. 不再主动计算 / 传递 BLOCK_D 与 num_warps，由 autotune 自动注入。
"""

from __future__ import annotations


import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_layer_norm_fwd_kernel import _layer_norm_fwd


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _get_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("_layer_norm_fwd requires an NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


@register("triton_layer_norm._layer_norm_fwd")
@register("triton_layer_norm_fwd")
class LayerNormFwdAutotuneApi(BaseApi):
    """_layer_norm_fwd kernel 的 Autotune API 封装。"""

    def __init__(self):
        self._device = None
        self._x = None
        self._y = None
        self._mean = None
        self._rstd = None
        self._d = None
        self._eps = None
        self._stride_x = None
        self._stride_y = None
        self._grid = None

    def init_by_input_data(self, input_data: InputDataset):
        """根据 ATK InputDataset 组装 kernel 入参（不计时）。"""
        # ── 1. 读取输入参数 ──
        x_input = input_data.kwargs.get("X")
        eps_input = input_data.kwargs.get("eps")

        # ── 2. 输入校验 ──
        if not isinstance(x_input, torch.Tensor):
            raise ValueError("X must be a tensor")
        if x_input.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got {x_input.ndim}D")
        if x_input.shape[0] == 0:
            raise ValueError("X must have non-zero row count")

        # ── 3. 获取设备 ──
        device = _get_device()
        self._device = device

        # ── 4. 计算 kernel 参数 ──
        # double_benchmark 模式下 ATK 可能生成 fp64 输入；kernel 只支持
        # fp32/fp16/bf16，且 NPU 不支持 fp64 Cast。需在 CPU 上先降到 fp32。
        n, d = x_input.shape
        if x_input.dtype == torch.float64:
            x = x_input.to(torch.float32).contiguous()
        else:
            x = x_input.contiguous()
        x = x.to(device)
        dtype = x.dtype

        y = torch.empty((n, d), dtype=dtype, device=device).contiguous()
        mean = torch.empty((n,), dtype=torch.float32, device=device)
        rstd = torch.empty((n,), dtype=torch.float32, device=device)

        grid = (n,)

        # ── 5. eps：转成 float 供 kernel 使用 ──
        eps = float(eps_input) if eps_input is not None else 1e-5

        self._x = x
        self._y = y
        self._mean = mean
        self._rstd = rstd
        self._d = d
        self._eps = eps
        self._stride_x = x.stride(0)
        self._stride_y = y.stride(0)
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行 kernel（计时）。autotune 自动注入 BLOCK_D / num_warps。"""
        _synchronize(self._device)

        _layer_norm_fwd[self._grid](
            self._x,
            self._y,
            self._mean,
            self._rstd,
            self._d,
            self._eps,
            self._stride_x,
            self._stride_y,
            TRAINING=True,
            COMPUTE_MEAN_AND_RSTD=True,
        )

        _synchronize(self._device)

        return self._y, self._mean, self._rstd
