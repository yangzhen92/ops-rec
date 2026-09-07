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

"""triton_layer_norm_fwd 的 ATK shape 约束生成器。

该 kernel 对 X [N, D] 的每一行做 layer normalization。关键约束：
  1. X 与 Y 的 shape 必须一致：[N, D]
  2. Mean 与 Rstd 的 shape 必须一致：[N]
  3. Y 的 dtype 必须与 X 一致（输出写回 X 的 element type）
  4. Mean / Rstd 恒为 fp32（与原生 wrapper 的累加 buffer 一致）
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_layer_norm_fwd")
class TritonLayerNormFwdGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 _layer_norm_fwd 可执行的 shape 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = case_config.inputs[0]
        y = case_config.inputs[1]
        mean = case_config.inputs[2]
        rstd = case_config.inputs[3]

        # ── 1. 统一 N, D ──
        # X 的 shape 确定 N（行数）与 D（特征维）
        n = int(x.shape[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else 64
        d = int(x.shape[1]) if hasattr(x, 'shape') and len(x.shape) > 1 else 64

        # Y 与 X 同 shape
        y.shape = [n, d]
        # Mean / Rstd 为一维 [N]
        mean.shape = [n]
        rstd.shape = [n]

        # ── 2. dtype 对齐 ──
        if hasattr(x, 'dtype'):
            x_dtype = x.dtype
            # Y 与 X 同 dtype（归一化结果写回输入类型）
            if hasattr(y, 'dtype'):
                y.dtype = x_dtype
            # Mean / Rstd 恒为 fp32（kernel 内 fp32 累加，写入 buffer）
            if hasattr(mean, 'dtype'):
                mean.dtype = 'fp32'
            if hasattr(rstd, 'dtype'):
                rstd.dtype = 'fp32'

        return case_config
