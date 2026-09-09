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
group_norm_bwd_dwdb 算子参数约束生成器

作用：
1. 把 DW 和 DB 从一维 [GROUP_N] 变成二维 [GROUP_N, num_heads]
2. 确保 DW 和 DB 的 shape 一致
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_group_norm_bwd_dwdb")
class GroupNormBwdDwdbGenerator(CaseGenerator):
    """group_norm_bwd_dwdb 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        dw = case_config.inputs[0]
        db = case_config.inputs[1]
        num_heads_input = case_config.inputs[2]

        # 获取 num_heads 的值
        num_heads_val = 1
        if hasattr(num_heads_input, 'range_values'):
            rv = num_heads_input.range_values
            if isinstance(rv, list) and len(rv) == 2:
                valid_nh = [nh for nh in [1, 2, 4, 8] if rv[0] <= nh <= rv[1]]
                num_heads_val = valid_nh[0] if valid_nh else 1
            elif isinstance(rv, list) and len(rv) > 0:
                num_heads_val = int(rv[0])
            else:
                num_heads_val = int(rv)

        # 获取 DW 的 GROUP_N 值（第一维）
        group_n = 64
        if hasattr(dw, 'shape') and isinstance(dw.shape, list) and len(dw.shape) > 0:
            group_n = dw.shape[0]

        # 把 DW 从一维 [GROUP_N] 变成二维 [GROUP_N, num_heads]
        if hasattr(dw, 'shape'):
            dw.shape = [group_n, num_heads_val]

        # 把 DB 变成和 DW 一样的 shape
        if hasattr(db, 'shape'):
            db.shape = [group_n, num_heads_val]

        return case_config
