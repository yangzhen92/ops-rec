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

# generate_triton_jagged_jagged_bmm_reduce_sum.py
"""_jagged_jagged_bmm_reduce_sum 的 ATK 参数约束生成器。"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


_SMALL_DIM = (1, 16, 31, 32, 63, 64, 65)
_LARGE_DIM = (64, 65, 96, 127, 128, 256, 384, 512)
_SMALL_BATCH = (1, 2, 4, 8)
_LARGE_BATCH = (8, 16, 32)

_MAX_TOTAL_ELEMENTS = 1 << 24
_MAX_L = 65536

_BLOCK_M = 64
_BLOCK_N = 64
_MAX_CORE_DIM = 65535


def _nearest_dim(value: int, candidates: tuple[int, ...]) -> int:
    return min(candidates, key=lambda c: (abs(c - value), c))


def _select_batch_size(offset_length: int, m: int, n: int, is_small: bool) -> int:
    candidates = _SMALL_BATCH if is_small else _LARGE_BATCH
    seed = max(1, offset_length - 1) + m + n
    return candidates[seed % len(candidates)]


@GENERATOR_REGISTRY.register("generate_jagged_jagged_bmm_reduce_sum")
class JaggedJaggedBmmReduceSumGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 kernel 可执行的 B、L、M、N 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        seq_offsets, jagged_a, jagged_b, reduce_out = case_config.inputs

        requested_l = int(jagged_a.shape[0])
        requested_m = int(jagged_a.shape[1])
        requested_n = int(jagged_b.shape[1])
        requested_offset_length = int(seq_offsets.shape[0])

        is_small = requested_l <= 1024
        dim_candidates = _SMALL_DIM if is_small else _LARGE_DIM
        m = _nearest_dim(max(1, requested_m), dim_candidates)
        n = _nearest_dim(max(1, requested_n), dim_candidates)
        batch_size = _select_batch_size(requested_offset_length, requested_m, requested_n, is_small)

        batch_size = min(batch_size, requested_l)

        fixed_elements = batch_size * (m * n + n)
        max_l_by_memory = max(batch_size, (_MAX_TOTAL_ELEMENTS - fixed_elements) // (m + n))
        total_length = min(max(batch_size, requested_l), _MAX_L, max_l_by_memory)

        batch_size = min(batch_size, total_length)
        fixed_elements = batch_size * (m * n + n)
        max_l_by_memory = max(batch_size, (_MAX_TOTAL_ELEMENTS - fixed_elements) // (m + n))
        total_length = min(max(batch_size, requested_l), _MAX_L, max_l_by_memory)

        core_dim = _cdiv(m, _BLOCK_M) * _cdiv(n, _BLOCK_N) * batch_size
        while core_dim > _MAX_CORE_DIM and m > 1:
            m = max(1, m // 2)
            core_dim = _cdiv(m, _BLOCK_M) * _cdiv(n, _BLOCK_N) * batch_size
        while core_dim > _MAX_CORE_DIM and n > 1:
            n = max(1, n // 2)
            core_dim = _cdiv(m, _BLOCK_M) * _cdiv(n, _BLOCK_N) * batch_size
        while core_dim > _MAX_CORE_DIM and batch_size > 1:
            batch_size //= 2
            core_dim = _cdiv(m, _BLOCK_M) * _cdiv(n, _BLOCK_N) * batch_size

        jagged_a.shape = [total_length, m]
        jagged_b.shape = [total_length, n]
        reduce_out.shape = [batch_size, n]
        seq_offsets.shape = [batch_size + 1]

        if hasattr(jagged_a, "dtype"):
            jagged_b.dtype = jagged_a.dtype
            reduce_out.dtype = jagged_a.dtype
        return case_config


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
