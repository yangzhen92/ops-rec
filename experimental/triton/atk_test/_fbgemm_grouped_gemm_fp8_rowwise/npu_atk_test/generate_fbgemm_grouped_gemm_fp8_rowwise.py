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
# limitations under the License.
# ==============================================================================

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

_ALIGN = 8
_MIN_ROWS_PER_GROUP = 64


@GENERATOR_REGISTRY.register("generate__fbgemm_grouped_gemm_fp8_rowwise")
class _FbgemmGroupedGemmFp8RowwiseGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 3:
            return case_config

        x = case_config.inputs[0]
        w = case_config.inputs[1]
        m_sizes = case_config.inputs[2]

        M = self._dim(x, 0, default=1024)
        K = self._dim(x, 1, default=128)
        N = self._dim(w, 0, default=256)  # yaml 第一维写的是 N，不是 G*N
        G = self._dim(m_sizes, 0, default=4)

        K = self._align_down(K)
        N = self._align_down(N)
        G = max(1, G)

        M = max(M, G * _MIN_ROWS_PER_GROUP)

        self._set_shape(x, [M, K])
        self._set_shape(w, [G * N, K])
        self._set_shape(m_sizes, [G])

        if hasattr(x, "dtype") and hasattr(w, "dtype"):
            w.dtype = x.dtype

        return case_config

    @staticmethod
    def _align_down(value):
        aligned = (value // _ALIGN) * _ALIGN
        return aligned if aligned >= _ALIGN else _ALIGN

    @staticmethod
    def _dim(tensor, index, default):
        shape = getattr(tensor, "shape", None)
        if isinstance(shape, list) and len(shape) > index:
            value = shape[index]
            if isinstance(value, int) and value > 0:
                return value
        return default

    @staticmethod
    def _set_shape(tensor, target_shape):
        if isinstance(getattr(tensor, "shape", None), list):
            tensor.shape[:] = target_shape
        else:
            tensor.shape = target_shape
