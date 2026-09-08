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

# generate_group_norm_mul_dropout_fwd.py
"""_group_norm_mul_dropout_fwd 算子参数约束生成器。"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


def is_power_of_2(n):
    """判断 n 是否是 2 的幂次方"""
    return n > 0 and (n & (n - 1)) == 0


def get_next_power_of_2(n):
    """获取大于等于 n 的最小 2 的幂次方"""
    return 1 if n <= 0 else 1 << (n - 1).bit_length()


@GENERATOR_REGISTRY.register("generate_group_norm_mul_dropout_fwd")
class BwdPreprocessDoODotGenerator(CaseGenerator):
    """使 X/U、W/Bias 的形状及四个输入的 dtype 满足算子约束。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """返回满足输入关系约束的用例。"""
        # 获取输入参数
        X = case_config.inputs[0]
        U = case_config.inputs[1]
        W = case_config.inputs[2]
        Bias = case_config.inputs[3]

        # 获取 X 的 shape
        X_shape = list(X.shape) if hasattr(X, "shape") else [1, 1, 1]
        X_dtype = X.dtype if hasattr(X, "dtype") else "fp32"

        # 约束5：D（第三维）必须是 2 的幂次方
        if len(X_shape) >= 3:
            D = X_shape[2]
            # 如果 D 不是 2 的幂次方，调整为最近的 2 的幂次方
            if not is_power_of_2(D):
                X_shape[2] = get_next_power_of_2(D)
                # 修改 X 的第三维
                X.shape = X_shape

        # 设置 U 的 shape 与 X 一致
        if U is not None and hasattr(U, "shape"):
            U.shape = X.shape

        # 设置 W 的 shape 与 X 的第二维一致
        if W is not None and hasattr(W, "shape"):
            W.shape = [X.shape[1]]

        # 设置 Bias 的 shape 与 X 的第二维一致
        if Bias is not None and hasattr(Bias, "shape"):
            Bias.shape = [X.shape[1]]

        # 设置 U, W, Bias 的 dtype 都与 X 保持一致
        if U is not None and hasattr(U, "dtype"):
            U.dtype = X_dtype
        if W is not None and hasattr(W, "dtype"):
            W.dtype = X_dtype
        if Bias is not None and hasattr(Bias, "dtype"):
            Bias.dtype = X_dtype

        return case_config
