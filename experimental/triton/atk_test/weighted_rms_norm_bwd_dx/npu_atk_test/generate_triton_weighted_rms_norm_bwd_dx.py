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
# generate_triton_weighted_rms_norm_bwd_dx.py
"""_weighted_rms_norm_bwd_dx 的 ATK shape 约束生成器。

该 kernel 对 RMSNorm 反向传播的输入梯度 DY 求 DX，并累加 partial weight gradient。
关键约束：
  1. X 与 DY 的 shape 必须一致：[N, D]
  2. W 的 shape 为 [D]（与 X 的第二维一致）
  3. Rstd 的 shape 为 [N]（与 X 的第一维一致）
  4. X / DY / W 的 dtype 必须一致
  5. Rstd 恒为 fp32（kernel 内 fp32 载荷）
  6. DX / DW / Lock 为 api 侧分配的内部 buffer，不在此定义
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_weighted_rms_norm_bwd_dx")
class WeightedRmsNormBwdDxGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 _weighted_rms_norm_bwd_dx 可执行的 shape 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = case_config.inputs[0]  # X
        dy = case_config.inputs[1]  # DY
        w = case_config.inputs[2]  # W
        rstd = case_config.inputs[3]  # Rstd

        # ── 1. 统一 N, D ──
        n = int(x.shape[0]) if hasattr(x, 'shape') and len(x.shape) > 0 else 64
        d = int(x.shape[1]) if hasattr(x, 'shape') and len(x.shape) > 1 else 64

        # NPU UB 限制：BLOCK_D = next_pow2(D) 必须 <= 8192，否则多个 16384 宽的
        # fp32 中间 buffer 同时驻留会 ub overflow（需 352KB > 可用 248KB）。
        # 若 D > 8192，裁剪到 8192（BLOCK_D=8192 可完整覆盖且不溢出）。
        _MAX_BLOCK_D = 8192
        d = min(d, _MAX_BLOCK_D)

        # DY 与 X 同 shape [N, D]
        dy.shape = [n, d]
        # W 为一维 [D]
        w.shape = [d]
        # Rstd 为一维 [N]
        rstd.shape = [n]

        # ── 2. dtype 对齐 ──
        if hasattr(x, 'dtype'):
            x_dtype = x.dtype
            # DY 与 X 同 dtype（同宽梯度）
            if hasattr(dy, 'dtype'):
                dy.dtype = x_dtype
            # W 与 X 同 dtype（kernel 内读取后转 fp32 参与计算）
            if hasattr(w, 'dtype'):
                w.dtype = x_dtype
            # Rstd 恒为 fp32
            if hasattr(rstd, 'dtype'):
                rstd.dtype = 'fp32'

        return case_config
