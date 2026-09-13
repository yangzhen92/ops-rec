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

# generate_triton_group_norm_mul_dropout_bwd_dx_du.py

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


_MAX_LINEAR_DIM = 4096
_MAX_TOTAL_ELEMENTS = 1 << 24
_HEAD_CANDIDATES = (32, 16, 8, 4, 2, 1)


@GENERATOR_REGISTRY.register("generate_triton_group_norm_mul_dropout_bwd_dx_du")
class GroupNormMulDropoutBwdDxDuGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x, u, weight, bias, dy = case_config.inputs[:5]
        x_shape = list(x.shape) if hasattr(x, "shape") else [1, 1]
        rows = max(1, int(x_shape[0]))
        inner = max(1, min(int(x_shape[-1]), _MAX_LINEAR_DIM))
        rows = min(rows, max(1, _MAX_TOTAL_ELEMENTS // inner))
        shape = [rows, inner]

        num_heads = next(heads for heads in _HEAD_CANDIDATES if inner % heads == 0)
        dtype = x.dtype if hasattr(x, "dtype") else "fp32"

        x.shape = shape
        for tensor in (u, dy):
            if tensor is not None and hasattr(tensor, "shape"):
                tensor.shape = shape
        for tensor in (weight, bias):
            if tensor is not None and hasattr(tensor, "shape"):
                tensor.shape = [num_heads]
        for tensor in (u, weight, bias, dy):
            if tensor is not None and hasattr(tensor, "dtype"):
                tensor.dtype = dtype

        return case_config
