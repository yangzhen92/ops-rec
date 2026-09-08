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
customized_gate_fwd_cublas_nhead 算子参数约束生成器

约束条件:
  - E = H * Dh (embedding_dim = num_heads * head_size)
  - proj_weights: (H, Din, head_size)   — 与 CUDA 接口一致
  - proj_biases: (H, Din)
  - gate_layer_weight: (E, Din)
  - gate_layer_bias: (E,)
  - rms_norm_weight: (Din,)  — 与 input 的 Din 一致
  - input: (B, Din)
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_customized_gate_fwd_cublas_nhead")
class CustomizedGateFwdCublasNheadGenerator(CaseGenerator):
    """customized_gate_fwd_cublas_nhead 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        input_tensor = case_config.inputs[0]  # (B, Din)
        gate_weight = case_config.inputs[1]  # (E, Din)
        gate_bias = case_config.inputs[2]  # (E,)
        proj_w = case_config.inputs[3]  # (H, Din, head_size)
        proj_b = case_config.inputs[4]  # (H, Din)
        rms_w = case_config.inputs[5]  # (Din,)

        # 1. 对齐 Din: input.Din = proj_w.shape[1] = proj_b.shape[-1] = rms_w.shape[0] = gate_weight.shape[-1]
        din = proj_w.shape[1]
        input_tensor.shape[-1] = din
        gate_weight.shape[-1] = din
        gate_weight.shape[-1] = din
        proj_w.shape[1] = din  # 同步更新 proj_weights 的 Din
        proj_b.shape[-1] = din
        rms_w.shape[0] = din

        # 2. 对齐 E: gate_weight.shape[0] = gate_bias.shape[0]
        e = gate_weight.shape[0]
        gate_bias.shape[0] = e

        # 3. 确保 E = H * Dh: 以 proj_w 的 H 和 head_size 为基准
        h = proj_w.shape[0]
        dh = proj_w.shape[2]
        target_e = h * dh
        gate_weight.shape[0] = target_e
        gate_bias.shape[0] = target_e

        # 4. 对齐 proj_bias H
        proj_b.shape[0] = h

        return case_config
