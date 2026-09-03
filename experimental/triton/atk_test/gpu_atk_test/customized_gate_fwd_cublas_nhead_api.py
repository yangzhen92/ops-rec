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
customized_gate_fwd_cublas_nhead 算子的 GPU 端 ATK API 执行文件

调用 cuda_hstu_ext.customized_gate_fwd_cublas_nhead（CUDA 算子）。

输入 shape 与 NPU（Triton）一致：
  proj_weights (H, Din, head_size) — API 直接透传，无需 permute
  proj_biases  (H, Din)
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

import cuda_hstu_ext


def _generate_tensor_by_shape(shape, dtype, device, seed=42):
    """根据 shape 确定性生成 tensor 值，与 NPU 侧对齐"""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=dtype, device='cpu')
    return t.to(device)


def _make_tensor(v, dtype, device):
    """统一将 ATK 传入的参数转为 tensor"""
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype)
    if isinstance(v, dict):
        shape = v.get('shape', v)
    elif isinstance(v, list):
        shape = v
    else:
        shape = v
    return _generate_tensor_by_shape(shape, dtype, device)


@register("customized_gate_fwd_cublas_nhead")
class GpuCustomizedGateFwdCublasNheadApi(BaseApi):
    """GPU 端 customized_gate_fwd_cublas_nhead 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）"""
        device = 'cuda'

        # ---- 从 input_data 获取原始参数 ----
        input_t = input_data.kwargs.get("input", None)
        gate_w_in = input_data.kwargs.get("gate_layer_weight", None)  # (E, Din)
        gate_b_in = input_data.kwargs.get("gate_layer_bias", None)  # (E,)
        proj_w_in = input_data.kwargs.get("proj_weights", None)  # (H, Din, head_size)
        proj_b_in = input_data.kwargs.get("proj_biases", None)  # (H, Din)
        rms_w_in = input_data.kwargs.get("rms_norm_weight", None)  # (Din,)
        eps_in = input_data.kwargs.get("eps", 1e-5)
        h_in = input_data.kwargs.get("input_gate_head", None)

        if input_t is None:
            raise ValueError("input not found in input_data")

        target_dtype = torch.bfloat16

        # ---- 统一生成 tensor（ATK json 传入的是 shape 描述，非实际数据） ----
        input_t = _make_tensor(input_t, target_dtype, device)
        gate_w = _make_tensor(gate_w_in, target_dtype, device)
        gate_b = _make_tensor(gate_b_in, target_dtype, device)
        proj_w = _make_tensor(proj_w_in, target_dtype, device)
        proj_b = _make_tensor(proj_b_in, target_dtype, device)
        rms_w = _make_tensor(rms_w_in, target_dtype, device)
        if proj_b.dim() == 3:
            proj_b = proj_b.squeeze(1)

        eps = (
            float(eps_in)
            if hasattr(eps_in, '__float__')
            else float(eps_in)
            if not isinstance(eps_in, float)
            else eps_in
        )

        # ---- 计算 input_gate_head（H） ----
        if h_in is not None:
            H = int(h_in)
        else:
            H = proj_w.shape[0]

        # ---- 确保连续性 ----
        for t in [input_t, gate_w, gate_b, proj_w, proj_b, rms_w]:
            if not t.is_contiguous():
                t = t.contiguous()

        # ---- 保存实例变量 ----
        self._device = device
        self._input = input_t
        self._gate_w = gate_w
        self._gate_b = gate_b
        self._proj_w = proj_w
        self._proj_b = proj_b
        self._rms_w = rms_w
        self._eps = eps
        self._H = H

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）"""
        torch.cuda.synchronize()

        output = cuda_hstu_ext.customized_gate_fwd_cublas_nhead(
            self._input,
            self._gate_w,
            self._gate_b,
            self._proj_w,
            self._proj_b,
            self._rms_w,
            self._eps,
            self._H,
        )

        torch.cuda.synchronize()

        return output
