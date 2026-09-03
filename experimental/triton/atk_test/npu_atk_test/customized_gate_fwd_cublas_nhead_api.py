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
customized_gate_fwd_cublas_nhead 算子的 ATK 自定义 API 执行文件

调用入口 customized_gate_fwd_cublas_nhead() 是一个三阶段融合算子：
  Kernel 1: torch.addmm → Gate GEMM (X @ W_gate^T + bias)
  Kernel 2: torch.baddbmm → Proj GEMM（bias 通过 baddbmm 注入）
  Kernel 3: Triton fused kernel → clamp+SiLU+sum_heads+gate_multiply+RMSNorm

输入 shape 与 CUDA 接口一致：
  proj_weights (H, Din, head_size) — API 直接透传，无需 permute
  proj_biases  (H, Din)            — 通过 baddbmm 注入
直接调用包装函数（已是最小粒度接口，不包含冗余预处理）。
"""

import os
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_customized_gate_fwd_cublas_nhead import customized_gate_fwd_cublas_nhead


def get_device():
    """获取可用的设备，优先 NPU；如果 FORCE_DEVICE=cpu 则强制使用 CPU"""
    force = os.environ.get("FORCE_DEVICE", "")
    if force == "cpu":
        return "cpu"
    if force == "npu":
        return "npu"
    if force == "cuda":
        return "cuda"
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _generate_tensor_by_shape(shape, dtype, device, seed=42):
    """
    根据 shape 确定性生成 tensor 值（不使用随机数）。
    GPU 和 NPU 使用同一份种子保证输入一致。
    """
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=dtype, device='cpu')
    return t.to(device)


@register("customized_gate_fwd_cublas_nhead")
class CustomizedGateFwdCublasNheadApi(BaseApi):
    """customized_gate_fwd_cublas_nhead 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）"""

        # ---- 从 input_data 获取原始参数 ----
        input_t = input_data.kwargs.get("input", None)
        gate_w_in = input_data.kwargs.get("gate_layer_weight", None)
        gate_b_in = input_data.kwargs.get("gate_layer_bias", None)
        proj_w_in = input_data.kwargs.get("proj_weights", None)
        proj_b_in = input_data.kwargs.get("proj_biases", None)
        rms_w_in = input_data.kwargs.get("rms_norm_weight", None)
        eps_in = input_data.kwargs.get("eps", 1e-5)

        if input_t is None:
            raise ValueError("input not found in input_data")

        # 统一使用 get_device() 确定运行设备，保证 GPU remote 节点上正确返回 'cuda'
        device = get_device()

        # ---- 转换到目标设备 ----
        if not isinstance(input_t, torch.Tensor):
            input_t = torch.tensor(input_t, dtype=torch.bfloat16, device=device)
        else:
            input_t = input_t.to(device)

        if not isinstance(gate_w_in, torch.Tensor):
            gate_w = _generate_tensor_by_shape(gate_w_in, torch.bfloat16, device)
        else:
            gate_w = gate_w_in.to(device)

        if not isinstance(gate_b_in, torch.Tensor):
            gate_b = _generate_tensor_by_shape(
                gate_b_in, torch.bfloat16 if not hasattr(gate_b_in, 'dtype') else gate_b_in.dtype, device
            )
        else:
            gate_b = gate_b_in.to(device)

        if not isinstance(proj_w_in, torch.Tensor):
            proj_w = _generate_tensor_by_shape(proj_w_in, torch.bfloat16, device)
        else:
            proj_w = proj_w_in.to(device)

        if not isinstance(proj_b_in, torch.Tensor):
            proj_b = _generate_tensor_by_shape(
                proj_b_in, torch.bfloat16 if not hasattr(proj_b_in, 'dtype') else proj_b_in.dtype, device
            )
        else:
            proj_b = proj_b_in.to(device)

        if not isinstance(rms_w_in, torch.Tensor):
            rms_w = _generate_tensor_by_shape(
                rms_w_in, torch.bfloat16 if not hasattr(rms_w_in, 'dtype') else rms_w_in.dtype, device
            )
        else:
            rms_w = rms_w_in.to(device)

        eps = (
            float(eps_in)
            if hasattr(eps_in, '__float__')
            else float(eps_in)
            if not isinstance(eps_in, float)
            else eps_in
        )

        # ---- 确保连续性 ----
        if not input_t.is_contiguous():
            input_t = input_t.contiguous()
        if not gate_w.is_contiguous():
            gate_w = gate_w.contiguous()
        if not gate_b.is_contiguous():
            gate_b = gate_b.contiguous()
        if not proj_w.is_contiguous():
            proj_w = proj_w.contiguous()
        if not proj_b.is_contiguous():
            proj_b = proj_b.contiguous()
        if not rms_w.is_contiguous():
            rms_w = rms_w.contiguous()

        # ---- 保存实例变量 ----
        self._device = device
        self._input = input_t
        self._gate_w = gate_w
        self._gate_b = gate_b
        self._proj_w = proj_w
        self._proj_b = proj_b
        self._rms_w = rms_w
        self._eps = eps

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        output = customized_gate_fwd_cublas_nhead(
            self._input,
            self._gate_w,
            self._gate_b,
            self._proj_w,
            self._proj_b,
            self._rms_w,
            eps=self._eps,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return output
