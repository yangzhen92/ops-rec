# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig
import random


@GENERATOR_REGISTRY.register("generate_triton_jagged_to_dense")
class JaggedToDenseGenerator(CaseGenerator):
    """jagged_to_dense 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        jagged_values = case_config.inputs[0]
        jagged_max_lengths = case_config.inputs[2]
        operation_function = case_config.inputs[4] if len(case_config.inputs) > 4 else None
        operation_value = case_config.inputs[5] if len(case_config.inputs) > 5 else None

        jagged_offsets = case_config.inputs[1]

        # jagged_offsets dtype 统一
        if isinstance(jagged_offsets, list) and len(jagged_offsets) > 0:
            first_dtype = None
            if hasattr(jagged_offsets[0], 'dtype'):
                first_dtype = jagged_offsets[0].dtype
            if first_dtype:
                for offset in jagged_offsets:
                    if hasattr(offset, 'dtype'):
                        offset.dtype = first_dtype

        # len(jagged_offsets[i]) < len(jagged_offsets[i+1])
        if isinstance(jagged_offsets, list):
            base = random.randint(2, 5)
            for offset in jagged_offsets:
                if hasattr(offset, 'shape'):
                    offset.shape = [base]
                    base = base * random.randint(2, 5)

        # jagged_max_lengths 与 jagged_offsets 数量一致
        num_jml = len(jagged_max_lengths) if isinstance(jagged_max_lengths, list) else 0
        num_jo = len(jagged_offsets) if isinstance(jagged_offsets, list) else 0
        target_num = min(num_jo, num_jml) if num_jo > 0 and num_jml > 0 else max(num_jo, num_jml)
        if target_num > 0:
            if isinstance(jagged_offsets, list) and len(jagged_offsets) > target_num:
                jagged_offsets[:] = jagged_offsets[:target_num]
            if isinstance(jagged_max_lengths, list) and len(jagged_max_lengths) > target_num:
                jagged_max_lengths[:] = jagged_max_lengths[:target_num]

        # jagged_values.shape[0] >= jagged_offsets[-1].shape[0]
        min_total = jagged_offsets[-1].shape[0] * random.randint(2, 3)
        if jagged_values.shape[0] < min_total:
            jagged_values.shape = [min_total, jagged_values.shape[-1]]

        # 获取 operation_function 的值
        op_func_val = None
        if operation_function is not None:
            if hasattr(operation_function, 'range_values'):
                rv = operation_function.range_values
                if rv is not None and str(rv).lower() != 'null':
                    op_func_val = str(rv)

        # 当 operation_function 是 "add" 或 "mul" 时，将 operation_value 的 dtype 改成与 jagged_values 一致
        jv_dtype = None
        if hasattr(jagged_values, 'dtype'):
            jv_dtype = jagged_values.dtype
        if op_func_val is not None and op_func_val not in ['null', 'none', '']:
            if operation_value is not None and jv_dtype is not None:
                operation_value.dtype = jv_dtype

        return case_config
