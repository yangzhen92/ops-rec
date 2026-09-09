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

"""jagged_dense_bmm_broadcast_add_kernel 的 ATK 参数约束生成器。"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


_SMALL_MATRIX_DIMENSIONS = (1, 16, 31, 32, 63, 64, 65)
_LARGE_MATRIX_DIMENSIONS = (64, 65, 96, 127, 128, 256, 384, 512)
_SMALL_BATCH_SIZES = (1, 2, 4, 8)
_LARGE_BATCH_SIZES = (8, 16, 32)

_MAX_TOTAL_ELEMENTS = 1 << 24
_MAX_L = 65536

_BLOCK_M = 64
_BLOCK_N = 64
_MAX_CORE_DIM = 65535


def _nearest_dimension(value: int, candidates: tuple[int, ...]) -> int:
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def _select_batch_size(offset_length: int, requested_k: int, requested_n: int, is_small: bool) -> int:
    """按负载档位确定 B，同时保留 offsets 原始 shape 对分布的影响。"""
    candidates = _SMALL_BATCH_SIZES if is_small else _LARGE_BATCH_SIZES
    seed = max(1, offset_length - 1) + requested_k + requested_n
    return candidates[seed % len(candidates)]


def _estimate_max_seq_len(total_length: int, batch_size: int) -> int:
    """估算 max_seq_len = ceil(L / B)。"""
    return (total_length + batch_size - 1) // batch_size


@GENERATOR_REGISTRY.register("generate_jagged_dense_bmm_broadcast_add")
class JaggedDenseBmmBroadcastAddGenerator(CaseGenerator):
    """将 ATK 独立生成的 shape 修正为 kernel 可执行的 B、L、K、N 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        seq_offsets, jagged, dense, bias = case_config.inputs

        requested_l = int(jagged.shape[0])
        requested_k = int(jagged.shape[1])
        requested_n = int(dense.shape[2])
        requested_offset_length = int(seq_offsets.shape[0])

        is_small = requested_l <= 1024
        matrix_dimensions = _SMALL_MATRIX_DIMENSIONS if is_small else _LARGE_MATRIX_DIMENSIONS
        k = _nearest_dimension(max(1, requested_k), matrix_dimensions)
        n = _nearest_dimension(max(1, requested_n), matrix_dimensions)
        batch_size = _select_batch_size(requested_offset_length, requested_k, requested_n, is_small)

        fixed_elements = batch_size * (k * n + n)
        max_l_by_memory = max(1, (_MAX_TOTAL_ELEMENTS - fixed_elements) // (k + n))
        total_length = min(max(1, requested_l), _MAX_L, max_l_by_memory)
        batch_size = min(batch_size, total_length)

        fixed_elements = batch_size * (k * n + n)
        max_l_by_memory = max(batch_size, (_MAX_TOTAL_ELEMENTS - fixed_elements) // (k + n))
        total_length = min(max(batch_size, requested_l), _MAX_L, max_l_by_memory)

        # 核心约束校验：NPU coreDim = cdiv(N, BLOCK_N) * cdiv(max_seq_len, BLOCK_M) * B <= 65535
        max_seq_len = _estimate_max_seq_len(total_length, batch_size)
        core_dim = _cdiv(n, _BLOCK_N) * _cdiv(max_seq_len, _BLOCK_M) * batch_size
        while core_dim > _MAX_CORE_DIM and n > 1:
            n = max(1, n // 2)
            core_dim = _cdiv(n, _BLOCK_N) * _cdiv(max_seq_len, _BLOCK_M) * batch_size
        while core_dim > _MAX_CORE_DIM and total_length > batch_size:
            total_length = max(batch_size, total_length * 3 // 4)
            max_seq_len = _estimate_max_seq_len(total_length, batch_size)
            core_dim = _cdiv(n, _BLOCK_N) * _cdiv(max_seq_len, _BLOCK_M) * batch_size

        jagged.shape = [total_length, k]
        dense.shape = [batch_size, k, n]
        bias.shape = [batch_size, n]
        seq_offsets.shape = [batch_size + 1]

        if hasattr(jagged, "dtype"):
            dense.dtype = jagged.dtype
            bias.dtype = jagged.dtype
        return case_config


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
