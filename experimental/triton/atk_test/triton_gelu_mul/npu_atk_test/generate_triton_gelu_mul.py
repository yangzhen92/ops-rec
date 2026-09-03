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
triton_gelu_mul 算子参数约束生成器
主要工作：确保最后一维是偶数（2d）。
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_gelu_mul")
class TritonGeluMulGenerator(CaseGenerator):
    """triton_gelu_mul 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        input_cfg = case_config.inputs[0]

        # 核心约束：最后一维必须是偶数（2d）
        if input_cfg.shape and len(input_cfg.shape) >= 1:
            last_dim = input_cfg.shape[-1]
            if last_dim % 2 != 0:
                input_cfg.shape[-1] = last_dim + 1

        return case_config
