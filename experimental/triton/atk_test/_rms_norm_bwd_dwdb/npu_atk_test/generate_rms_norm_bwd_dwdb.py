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
_rms_norm_bwd_dwdb 的 ATK shape 约束生成器。

该 kernel 对 _weighted_rms_norm_bwd_dx 产出的 partial weight gradient
做规约求和，输出最终的 dweight。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_rms_norm_bwd_dwdb")
class RMSNormBwdDwdbGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 kernel 可执行的 shape 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        dw = case_config.inputs[0]
        final_dw = case_config.inputs[1]

        # ── 1. 统一 D ──
        # DW 的第二维必须和 FINAL_DW 的第一维相同
        requested_d_dw = int(dw.shape[1]) if hasattr(dw, "shape") and len(dw.shape) > 1 else 64
        requested_d_final = int(final_dw.shape[0]) if hasattr(final_dw, "shape") and len(final_dw.shape) > 0 else 64
        d = max(requested_d_dw, requested_d_final)

        # ── 2. 确定 tile_num（N）──
        tile_num = int(dw.shape[0]) if hasattr(dw, "shape") and len(dw.shape) > 0 else 16

        # ── 3. 回写 shape ──
        dw.shape = [tile_num, d]
        final_dw.shape = [d]

        # ── 4. dtype 一致 ──
        if hasattr(dw, "dtype"):
            if hasattr(final_dw, "dtype"):
                final_dw.dtype = dw.dtype

        return case_config
