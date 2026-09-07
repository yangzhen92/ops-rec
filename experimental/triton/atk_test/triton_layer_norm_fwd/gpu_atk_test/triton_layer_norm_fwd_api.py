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
_layer_norm_fwd kernel ATK API 封装。

直接调用 _layer_norm_fwd，对输入 X [N, D] 的每一行做 layer normalization：
  grid = (N,)，每个 program 处理一行
  输出 Y [N, D]（归一化结果），并产出 Mean [N] / Rstd [N]（fp32）

Kernel 参数说明（见 triton_layer_norm.py:36-76）：
  - X: [N, D] 输入
  - Y: [N, D] 输出（归一化结果，与 X 同 dtype）
  - Mean: [N] 输出均值（fp32）
  - Rstd: [N] 输出 rstd（fp32）
  - D: 特征维
  - eps: 归一化 epsilon
  - stride_x / stride_y: X/Y 的行 stride
  - TRAINING: constexpr，True 时写 Mean/Rstd
  - BLOCK_D: constexpr，由 D 和 x.element_size() 计算
  - COMPUTE_MEAN_AND_RSTD: constexpr，True 时在 kernel 内计算 mean/rstd
"""

from __future__ import annotations


import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_layer_norm import _layer_norm_fwd


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
"""kernel 支持并参与精度对比的输入 dtype（X）。"""

# double_benchmark 模式：关闭 dump


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
@register("_layer_norm_fwd")
class LayerNormFwdApi(BaseApi):
    """_layer_norm_fwd kernel 的 API 封装。"""

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
        self._block_d = None
        self._num_warps = None
        self._grid = None

    # @staticmethod
    # def _make_case_id(x_input: torch.Tensor, eps) -> str:
    #     """根据输入值构造跨后端一致的 case ID，用于 dump 目录命名。

    #     NPU/GPU 使用相同输入时会生成相同 ID，便于对比两侧结果。
    #     """
    #     flat = x_input.cpu().flatten().float()
    #     avg = flat[:10].mean().item()
    #     return f"X{avg:.6f}_eps{float(eps):.6g}"

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
        # fp32/fp16/bf16，且 NPU 不支持 fp64 Cast。需在 CPU 上先降到 fp32，
        # 再移到目标设备（kernel 的输出 dtype / BLOCK_D 也基于降级后的 dtype）。
        n, d = x_input.shape
        if x_input.dtype == torch.float64:
            x = x_input.to(torch.float32).contiguous()
        else:
            x = x_input.contiguous()
        x = x.to(device)
        dtype = x.dtype
        element_size = x.element_size()

        # 与原生 wrapper 一致的 BLOCK_D / num_warps 计算
        #   MAX_FUSED_SIZE = 65536 // element_size()
        #   BLOCK_D = min(MAX_FUSED_SIZE, next_power_of_2(D))
        #   num_warps = min(max(BLOCK_D // 256, 1), 8)
        max_fused_size = 65536 // element_size
        block_d = min(max_fused_size, triton.next_power_of_2(d))
        num_warps = min(max(block_d // 256, 1), 8)

        # 输出 buffer：Y [N, D] 与 X 同 dtype；Mean/Rstd [N] 恒为 fp32
        y = torch.empty((n, d), dtype=dtype, device=device).contiguous()
        mean = torch.empty((n,), dtype=torch.float32, device=device)
        rstd = torch.empty((n,), dtype=torch.float32, device=device)

        # grid = (N,)
        grid = (n,)

        # ── 5. eps：转成 float 供 kernel 使用 ──
        eps = float(eps_input) if eps_input is not None else 1e-5

        # ── 6. 保存实例变量 ──
        self._x = x
        self._y = y
        self._mean = mean
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
            BLOCK_D=self._block_d,
            COMPUTE_MEAN_AND_RSTD=True,
            num_warps=self._num_warps,
        )

        _synchronize(self._device)

        return self._y, self._mean, self._rstd
