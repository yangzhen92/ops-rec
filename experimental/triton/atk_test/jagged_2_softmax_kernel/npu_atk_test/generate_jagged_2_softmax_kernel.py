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
jagged_2_softmax_kernel 算子参数约束生成器

作用：
1. 按 B 确定性生成 seq_len（与 API 逐字一致），设置 input_x=[Σseq²]、三个 offsets=[B+1]
2. 注入 offset 值仅为落盘可读性，运行时由 API 重建
"""

import random
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


def _hash_range(i, lo, hi, salt=0):
    """确定性 LCG 哈希：把索引 i 映射到 [lo, hi]（GPU/NPU 一致，无 random）"""
    x = ((i + 1) * 2654435761 + salt * 40503) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0x7FFFFFFF
    return lo + x % (hi - lo + 1)


def _deterministic_seq_lengths(B, lo=16, hi=256):
    """确定性生成 B 个 seq_len（种子仅由 B 决定，与 API 中实现逐字一致）"""
    if B <= 0:
        return []
    first_len = _hash_range(0, lo, hi, salt=(B * 2654435761) & 0x7FFFFFFF)
    lengths = [first_len]
    for i in range(1, B):
        lengths.append(_hash_range(i, lo, hi, salt=first_len))
    return lengths


@GENERATOR_REGISTRY.register("generate_jagged_2_softmax_kernel")
class Jagged2SoftmaxGenerator(CaseGenerator):
    """jagged_2_softmax_kernel 算子的输入生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """生成自洽的jagged方阵输入"""
        if len(case_config.inputs) < 4:
            return case_config

        input_x = case_config.inputs[0]
        offsets_row = case_config.inputs[1]
        offsets_col = case_config.inputs[2]
        offsets_overall = case_config.inputs[3]

        B = self._resolve_B(offsets_overall)
        lengths_seq = self._generate_seq_lengths(B)
        lengths_overall = [s * s for s in lengths_seq]
        total_overall = sum(lengths_overall)

        # 设置shape
        if hasattr(input_x, 'shape'):
            input_x.shape = [total_overall]
        if hasattr(offsets_row, 'shape'):
            offsets_row.shape = [B + 1]
        if hasattr(offsets_col, 'shape'):
            offsets_col.shape = [B + 1]
        if hasattr(offsets_overall, 'shape'):
            offsets_overall.shape = [B + 1]

        # 注入 offset 值仅为落盘可读性（ATK outlier_values 不覆盖 tensor 值，运行时由 API 重建）
        offsets_row.outlier_values = self._cumsum(lengths_seq)
        offsets_col.outlier_values = self._cumsum(lengths_seq)
        offsets_overall.outlier_values = self._cumsum(lengths_overall)

        return case_config

    def _resolve_B(self, offset_tensor):
        """从offset长度反推B"""
        if hasattr(offset_tensor, 'shape') and offset_tensor.shape:
            offset_len = offset_tensor.shape[0]
            if offset_len > 1:
                return offset_len - 1
        if hasattr(offset_tensor, 'dim_values') and offset_tensor.dim_values:
            offset_len = random.choice(offset_tensor.dim_values)
            if offset_len > 1:
                return offset_len - 1
        # 兜底：对齐 yaml dim_values 的 B 集合（1/3/5/7/10/15）
        return random.choice([1, 3, 5, 7, 10, 15])

    def _generate_seq_lengths(self, B):
        """确定性生成 B 个 seq_len，委托模块级 _deterministic_seq_lengths（见上方）"""
        return _deterministic_seq_lengths(B)

    def _cumsum(self, lengths):
        """cumsum生成offset"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return offsets
