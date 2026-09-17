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

# generate_triton_jagged_reduce_sum.py
"""jagged_reduce_sum 的 ATK shape 约束生成器。"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


# 与 kernel 的 BLOCK_D 分支对应，保留 63/64/65 尾块并扩展到 8192。
_D_CANDIDATES = (16, 31, 63, 64, 65, 128, 256, 512, 1024, 2048, 4096, 8192)
_B_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# Jagged 与 Out 的合计元素数上限。fp32 下约为 128 MiB，不含框架额外开销。
_MAX_TOTAL_ELEMENTS = 1 << 24
_MAX_L = 65536


def _nearest_supported_d(value: int) -> int:
    """把 YAML 采样维度映射到含大性能 shape 的 D 候选集。"""
    return min(_D_CANDIDATES, key=lambda candidate: (abs(candidate - value), candidate))


def _select_batch_size(offset_length: int, requested_l: int, requested_d: int) -> int:
    """用小的 seq_offsets 原生 shape 池稳定通过 ATK，再确定性扩展 batch 覆盖面。"""
    base_batch = max(1, offset_length - 1)
    if base_batch <= 16:
        base_index = (base_batch.bit_length() - 1) if base_batch & (base_batch - 1) == 0 else 0
    else:
        base_index = 0
    # 只依赖 shape，不使用随机数；同一条 case 在 NPU/GPU 两端会得到相同的 B。
    candidate_index = (base_index + requested_l + requested_d) % len(_B_CANDIDATES)
    return _B_CANDIDATES[candidate_index]


@GENERATOR_REGISTRY.register("generate_jagged_reduce_sum")
class JaggedReduceSumGenerator(CaseGenerator):
    """只调整 shape，使每一个生成的配置都能被 API 安全执行。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        jagged = case_config.inputs[0]
        seq_offsets = case_config.inputs[1]

        # seq_offsets 的长度定义 batch 数：B = len(seq_offsets) - 1。
        # 因而长度至少为 2，保证 B >= 1，避免零 grid 的无效 kernel launch。
        # Jagged 必须是 [L, D]，D 不能为 0。D 的候选值覆盖 BLOCK_D 的分支边界。
        requested_l = int(jagged.shape[0])
        requested_d = int(jagged.shape[1])
        d = _nearest_supported_d(max(1, requested_d))

        requested_offset_length = int(seq_offsets.shape[0])
        batch_size = _select_batch_size(requested_offset_length, requested_l, requested_d)
        seq_offsets.shape = [batch_size + 1]

        # 主测试集排除 L=0；同时限制 (L + B) * D，覆盖 Jagged 和 Out 的主要内存占用。
        max_l_by_memory = max(1, (_MAX_TOTAL_ELEMENTS // d) - batch_size)
        jagged_length = min(max(1, requested_l), _MAX_L, max_l_by_memory)
        jagged.shape = [jagged_length, d]

        return case_config
