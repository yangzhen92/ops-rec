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
triton_jagged_jagged_elementwise_arithmetic_ops 算子参数约束生成器
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_jagged_jagged_elementwise_arithmetic_ops")
class TritonJaggedJaggedElementwiseArithmeticOpsGenerator(CaseGenerator):
    """triton_jagged_jagged_elementwise_arithmetic_ops 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """
        用例参数约束修改入口

        约束：
        1. x_values 和 y_values 必须有相同的 shape
        2. x_values 和 y_values 的 dtype 必须一致
        """
        # 获取输入参数
        if len(case_config.inputs) < 2:
            return case_config

        x_values = case_config.inputs[0]
        y_values = case_config.inputs[1]

        # 约束1: y_values 的 shape 与 x_values 一致
        if hasattr(x_values, 'shape') and hasattr(y_values, 'shape'):
            y_values.shape = x_values.shape

        # 约束2: y_values 的 dtype 与 x_values 一致
        if hasattr(x_values, 'dtype') and hasattr(y_values, 'dtype'):
            y_values.dtype = x_values.dtype

        return case_config
