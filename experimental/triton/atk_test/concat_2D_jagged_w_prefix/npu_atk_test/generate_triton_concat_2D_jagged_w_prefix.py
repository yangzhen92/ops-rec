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
# generate_triton_concat_2D_jagged_w_prefix.py
"""concat_2D_jagged_w_prefix kernel 的 ATK shape 约束生成器。"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


# D 维候选：覆盖 BLOCK_D = next_power_of_2(D) 的边界（1/2/4/8/16/32/64/128 等）
_D_CANDIDATES = (1, 2, 15, 16, 31, 32, 63, 64, 65, 128, 256, 512, 1024, 2048, 4096, 8192)
# batch 候选：覆盖 B=1 到 1024
_B_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
# 总元素数上限（Out + ValuesA + ValuesB 约 128 MiB）
_MAX_TOTAL_ELEMENTS = 1 << 24
# 单序列长度上限
_MAX_SEQ_LEN = 65536
# coreDim 上限（二维 grid: max_seq_len × B）
_MAX_CORE_DIM = 65535


def _nearest_supported_d(value: int) -> int:
    """将 YAML 采样维度映射到 D 候选集。"""
    return min(_D_CANDIDATES, key=lambda candidate: (abs(candidate - value), candidate))


def _select_batch_size(offset_length: int, requested_d: int) -> int:
    """按确定性规则选择 batch 大小。"""
    base_index = max(1, offset_length - 1)
    seed = (base_index + requested_d) % len(_B_CANDIDATES)
    return _B_CANDIDATES[seed]


@GENERATOR_REGISTRY.register("generate_concat_2D_jagged_w_prefix")
class Concat2DJaggedWPrefixGenerator(CaseGenerator):
    """将 YAML 采样 shape 修正为 kernel 可执行的 B、L_A、L_B、D 组合。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        seq_offsets, values_a, values_b = case_config.inputs[:3]

        requested_la = int(values_a.shape[0])
        requested_lb = int(values_b.shape[0])
        requested_d = int(values_a.shape[1])
        requested_offset_length = int(seq_offsets.shape[0])

        d = _nearest_supported_d(max(1, requested_d))
        batch_size = _select_batch_size(requested_offset_length, requested_d)

        # 公共的 D 维
        values_a.shape[1] = d
        values_b.shape[1] = d

        # 每 batch 最多一个 token 的保证：L >= B
        max_total = _MAX_TOTAL_ELEMENTS // d
        # Out = (L_A + L_B, D), ValuesA = (L_A, D), ValuesB = (L_B, D)
        # 总元素数 = (L_A + L_B) * D + L_A * D + L_B * D = 2 * (L_A + L_B) * D
        # 所以 L_A + L_B <= _MAX_TOTAL_ELEMENTS // (2 * D)
        max_seq_total = max_total // 2

        # L_A 和 L_B 分别调整: 每个至少 batch_size 个 token，总和不超过限制
        la = min(max(batch_size, requested_la), _MAX_SEQ_LEN)
        lb = min(max(batch_size, requested_lb), _MAX_SEQ_LEN)

        # 总和超过上限时按比例收缩
        if la + lb > max_seq_total:
            scale = max_seq_total / (la + lb)
            la = max(batch_size, int(la * scale))
            lb = max(batch_size, int(lb * scale))

        # 总和仍不够则优先满足 A
        if la + lb > max_seq_total:
            la = max(batch_size, max_seq_total - batch_size)
            lb = batch_size

        values_a.shape[0] = la
        values_b.shape[0] = lb
        seq_offsets.shape = [batch_size + 1]

        # 三输入 dtype 一致
        if hasattr(values_a, "dtype"):
            values_b.dtype = values_a.dtype

        return case_config
