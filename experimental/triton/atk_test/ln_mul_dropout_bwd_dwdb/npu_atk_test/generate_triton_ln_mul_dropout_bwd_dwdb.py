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


_D_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


_MAX_TOTAL_ELEMENTS = 1 << 22
_MAX_D = 32768
_MAX_TILE_NUM = 1024


def _nearest_supported_d(value: int) -> int:
    return min(_D_CANDIDATES, key=lambda candidate: (abs(candidate - value), candidate))


@GENERATOR_REGISTRY.register("generate_ln_mul_dropout_bwd_dwdb")
class LnMulDropoutBwdDwdbGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        dw = case_config.inputs[0]
        db = case_config.inputs[1]
        final_dw = case_config.inputs[2]
        final_db = case_config.inputs[3]

        requested_d_dw = int(dw.shape[1]) if hasattr(dw, 'shape') and len(dw.shape) > 1 else 64
        requested_d_db = int(db.shape[1]) if hasattr(db, 'shape') and len(db.shape) > 1 else 64
        requested_d_final = int(final_dw.shape[0]) if hasattr(final_dw, 'shape') and len(final_dw.shape) > 0 else 64
        requested_d = max(requested_d_dw, requested_d_db, requested_d_final)
        d = min(_nearest_supported_d(max(1, requested_d)), _MAX_D)

        requested_tile_num_dw = int(dw.shape[0]) if hasattr(dw, 'shape') and len(dw.shape) > 0 else 16
        requested_tile_num_db = int(db.shape[0]) if hasattr(db, 'shape') and len(db.shape) > 0 else 16
        tile_num = max(1, min(max(requested_tile_num_dw, requested_tile_num_db), _MAX_TILE_NUM))

        max_tile_by_memory = max(1, _MAX_TOTAL_ELEMENTS // d) if d > 0 else _MAX_TILE_NUM
        tile_num = min(tile_num, max_tile_by_memory)

        dw.shape = [tile_num, d]
        db.shape = [tile_num, d]
        final_dw.shape = [d]
        final_db.shape = [d]

        if hasattr(dw, 'dtype'):
            db.dtype = dw.dtype
            if hasattr(final_dw, 'dtype'):
                final_dw.dtype = dw.dtype
            if hasattr(final_db, 'dtype'):
                final_db.dtype = dw.dtype

        return case_config
