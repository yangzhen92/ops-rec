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

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


_D_CANDIDATES = (1, 2, 15, 16, 31, 32, 63, 64, 65, 128, 256, 512, 1024, 2048, 4096)


_MAX_TOTAL_ELEMENTS = 1 << 24
_MAX_N = 65536


def _nearest_supported_d(value: int) -> int:
    return min(_D_CANDIDATES, key=lambda candidate: (abs(candidate - value), candidate))


@GENERATOR_REGISTRY.register("generate_ln_mul_dropout_bwd_dx_du")
class LnMulDropoutBwdDxDuGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x_input = case_config.inputs[0]
        u_input = case_config.inputs[1]
        dy_input = case_config.inputs[2]

        requested_d_x = int(x_input.shape[1])
        requested_d_u = int(u_input.shape[1])
        requested_d_dy = int(dy_input.shape[1])
        requested_d = max(requested_d_x, requested_d_u, requested_d_dy)
        d = _nearest_supported_d(max(1, requested_d))

        requested_n_x = int(x_input.shape[0])
        requested_n_u = int(u_input.shape[0])
        requested_n_dy = int(dy_input.shape[0])
        requested_n = max(requested_n_x, requested_n_u, requested_n_dy)

        max_n_by_memory = max(1, _MAX_TOTAL_ELEMENTS // d) if d > 0 else _MAX_N
        n = min(max(1, requested_n), _MAX_N, max_n_by_memory)

        x_input.shape = [n, d]
        u_input.shape = [n, d]
        dy_input.shape = [n, d]

        if len(case_config.inputs) > 3:
            w_input = case_config.inputs[3]
            if w_input is not None:
                w_input.shape = [d]

        if len(case_config.inputs) > 4:
            b_input = case_config.inputs[4]
            if b_input is not None:
                b_input.shape = [d]

        return case_config
