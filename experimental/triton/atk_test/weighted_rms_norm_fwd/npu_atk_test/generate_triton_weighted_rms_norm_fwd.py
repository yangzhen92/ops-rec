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
# generate_triton_weighted_rms_norm_fwd.py
"""ATK shape constraint generator for weighted RMS normalization.

该 kernel 对 X [N, D] 的每一行做 RMS Norm 并乘上 weight。关键约束：
  1. X 与 Y 的 shape 必须一致：[N, D]
  2. W 的 shape 为 [D]（与 X 的第二维一致）
  3. Rstd 的 shape 为 [N]（与 X 的第一维一致）
  4. Y 的 dtype 必须与 X 一致（输出写回 X 的 element type）
  5. Rstd 恒为 fp32（与原 wrapper 里 rstd buffer 一致，均值为 rstd）
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_weighted_rms_norm_fwd")
class WeightedRmsNormFwdGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 _weighted_rms_norm_fwd 可执行的 shape 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = case_config.inputs[0]
        w = case_config.inputs[1]
        y = case_config.inputs[2]
        rstd = case_config.inputs[3]

        n = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 64
        d = int(x.shape[1]) if hasattr(x, "shape") and len(x.shape) > 1 else 64

        y.shape = [n, d]
        w.shape = [d]
        rstd.shape = [n]

        if hasattr(x, "dtype"):
            x_dtype = x.dtype
            if hasattr(y, "dtype"):
                y.dtype = x_dtype
            if hasattr(w, "dtype"):
                w.dtype = x_dtype
            if hasattr(rstd, "dtype"):
                rstd.dtype = "fp32"

        return case_config
