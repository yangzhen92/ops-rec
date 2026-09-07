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
 
"""_add_embeddings_bwd_kernel 的 ATK shape 约束生成器。

关键约束：
  1. In 为 2D [jagged_size, D]。
  2. num_keys（Out 的 key 数）必须 <= jagged_size，保证每个 key
     至少被一个 token 寻址（否则部分输出行永远为 0，虽不崩溃但覆盖面低）。
  3. D 由 API 通过 BLOCK_D = next_power_of_2(D) 编译，生成器不做截断；
     仅保证 D 为内存可承受的有限值（YAML 上限 8192）。

该生成器只调整 shape，真正的 KeyInds/ValueInds 推导与 Out 构造在 API 的
init_by_input_data 中执行（与教程 _layer_norm_bwd_dx 的 Mean/Rstd 推导一致）。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_add_embeddings_bwd_kernel")
class AddEmbeddingsBwdKernelGenerator(CaseGenerator):
    """只调整 shape，使每个生成的配置都能被 API 安全执行。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        in_input = case_config.inputs[0]
        num_keys_input = case_config.inputs[1] if len(case_config.inputs) > 1 else None

        # ── 1. 解析 num_keys（确定性取第一个候选值）──
        num_keys = 16
        if num_keys_input is not None and hasattr(num_keys_input, "range_values"):
            rv = num_keys_input.range_values
            if isinstance(rv, (int, float)):
                num_keys = int(rv)
            elif isinstance(rv, list) and len(rv) > 0:
                first = rv[0]
                if isinstance(first, (list, tuple)):
                    num_keys = int(first[0])
                else:
                    num_keys = int(first)
            elif isinstance(rv, str):
                num_keys = 16

        num_keys = max(1, num_keys)

        # ── 2. 解析 In 的采样 shape ──
        in_shape = list(in_input.shape) if in_input.shape else [1024, 1024]
        jagged_size = int(in_shape[0]) if len(in_shape) > 0 else 1024
        d = int(in_shape[1]) if len(in_shape) > 1 else 1024

        # ── 3. 约束：jagged_size >= num_keys，D 至少为 1 ──
        jagged_size = max(1, jagged_size, num_keys)
        d = max(1, d)

        in_input.shape = [jagged_size, d]

        return case_config