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

# generate__bwd_preprocess_do_o_dot.py
"""
_bwd_preprocess_do_o_dot 算子参数约束生成器

参数约束说明：
1. Out 和 do 的 shape 必须完全一致：[B, T, D]
2. Out 和 do 的 dtype 必须一致
3. D（第三维）必须是 2 的幂次方，因为 triton arange 要求范围是 2 的幂次方
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


def is_power_of_2(n):
    """判断 n 是否是 2 的幂次方"""
    return n > 0 and (n & (n - 1)) == 0


def get_next_power_of_2(n):
    """获取大于等于 n 的最小 2 的幂次方"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@GENERATOR_REGISTRY.register("generate_bwd_preprocess_do_o_dot")
class BwdPreprocessDoODotGenerator(CaseGenerator):
    """
    _bwd_preprocess_do_o_dot 算子参数约束生成器

    参数约束：
    1. Out 和 do 的 shape 必须完全一致
    2. Out 和 do 的 dtype 必须一致
    3. D（第三维）必须是 2 的幂次方
    """

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """
        用例参数约束修改入口
        :param case_config: 生成的用例信息
        :return: 返回修改后符合参数间约束关系的用例
        """
        # 获取输入参数
        Out = case_config.inputs[0]
        do = case_config.inputs[1]

        # 获取 Out 的 shape
        out_shape = list(Out.shape) if hasattr(Out, "shape") else [1, 1, 1]

        # 约束3：D（第三维）必须是 2 的幂次方
        if len(out_shape) >= 3:
            D = out_shape[2]
            # 如果 D 不是 2 的幂次方，调整为最近的 2 的幂次方
            if not is_power_of_2(D):
                out_shape[2] = get_next_power_of_2(D)
                # 修改 Out 的第三维
                Out.shape = out_shape

        # 设置 do 的 shape 与 Out 一致
        if do is not None and hasattr(do, "shape"):
            do.shape = Out.shape

        # 设置 do 的 dtype 与 Out 一致
        if do is not None and hasattr(do, "dtype"):
            do.dtype = Out.dtype

        return case_config
