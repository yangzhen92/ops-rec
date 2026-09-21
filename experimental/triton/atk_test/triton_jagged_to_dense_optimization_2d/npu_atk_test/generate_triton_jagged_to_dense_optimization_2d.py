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
triton_jagged_to_dense_optimization_2d 算子参数约束生成器
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_jagged_to_dense_optimization_2d")
class TritonJaggedToDenseOptimization2dGenerator(CaseGenerator):
    """triton_jagged_to_dense_optimization_2d 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        jagged_offsets = case_config.inputs[1]
        jagged_max_lengths = case_config.inputs[2]

        # 从 jagged_offsets 获取期望的 shape
        if hasattr(jagged_offsets, 'shape'):
            offset_shape = list(jagged_offsets.shape)
            outer_size = offset_shape[0] - 1  # offset 长度 = outer_size + 1

            # 约束1: 调整 jagged_offsets.shape = [outer_size + 1]
            jagged_offsets.shape = [outer_size + 1]

            # 约束2: 确保 jagged_max_lengths 是单元素列表（2D 情况）
            if hasattr(jagged_max_lengths, 'range_values'):
                rv = jagged_max_lengths.range_values
                if isinstance(rv, list) and len(rv) == 2:
                    max_len = rv[1]
                else:
                    max_len = 8
            else:
                max_len = 8

            # 将 jagged_max_lengths 设置为单元素列表
            jagged_max_lengths.range_values = [max_len]

        return case_config
