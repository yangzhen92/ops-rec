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

# generate_triton_layer_norm_bwd_dx.py
"""_layer_norm_bwd_dx 的 ATK shape 约束生成器。

关键约束：
  1. X 与 DY 必须同形 [N, D]（kernel 按行广播加法）。
  2. BLOCK_D = next_pow2(D) 必须覆盖整个特征维 D（mask `cols < D`），
     否则后 (D - BLOCK_D) 列不被计算、输出不完整。
     去掉硬编码的块大小上限后，D 可覆盖到 8192（YAML dim_values 上限）；
     BLOCK_D 由 API 按上游规则 BLOCK_D = min(65536//elem_size, next_pow2(D))
     计算，因此不再在此截断 D，D 仅受内存上限约束。
  3. 内存约束：X/DX/DY 三个同形 tensor 的元素总数不得超过
     _MAX_TOTAL_ELEMENTS，避免 OOM。

该生成器只调整 shape，真正的 Mean/Rstd 推导与 DX 构造在 API 的
init_by_input_data 中执行。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


# 总元素数上限，避免 OOM。X/DX/DY 三个同形 tensor，fp32 下
# 3 * _MAX_TOTAL_ELEMENTS * 4B ≈ 192MiB。
_MAX_TOTAL_ELEMENTS = 1 << 24
# 行数上限。
_MAX_N = 1 << 20


@GENERATOR_REGISTRY.register("generate_triton_layer_norm_bwd_dx")
class TritonLayerNormBwdDxGenerator(CaseGenerator):
    """只调整 shape，使每个生成的配置都能被 API 安全执行。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x_input = case_config.inputs[0]
        dy_input = case_config.inputs[1]

        x_shape = list(x_input.shape) if x_input.shape else [1024, 1024]
        n = int(x_shape[0]) if len(x_shape) > 0 else 1024
        d = int(x_shape[1]) if len(x_shape) > 1 else 1024

        # 保留 YAML 采样的 D（上限由 YAML 8192 决定），仅做最小保护。
        # 内存约束通过 N 的上限来满足：N * D <= _MAX_TOTAL_ELEMENTS。
        d_eff = max(1, d)

        n = max(1, n)
        max_n_by_memory = max(1, (int(_MAX_TOTAL_ELEMENTS) // d_eff) - 1)
        n = min(n, _MAX_N, max_n_by_memory)

        # X 与 DY 强制同形。
        x_input.shape = [n, d_eff]
        dy_input.shape = [n, d_eff]

        return case_config
