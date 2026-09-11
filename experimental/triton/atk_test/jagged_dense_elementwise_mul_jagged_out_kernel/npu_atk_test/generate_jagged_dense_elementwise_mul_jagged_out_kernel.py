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
jagged_dense_elementwise_mul_jagged_out_kernel 算子参数约束生成器

作用：
1. 生成各 batch 方阵边长 seq_len，设置 a_ptr=[Σseq²]、b_ptr=[max(seq), max(seq)] 方阵、offsets=seq² cumsum
2. 约束 a_ptr 与 b_ptr dtype 一致
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_jagged_dense_elementwise_mul_jagged_out_kernel")
class JaggedDenseElementwiseMulJaggedOutGenerator(CaseGenerator):
    """jagged_dense_elementwise_mul_jagged_out_kernel 算子的输入生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """生成自洽输入"""
        if len(case_config.inputs) < 5:
            return case_config

        a_ptr = case_config.inputs[0]
        b_ptr = case_config.inputs[1]
        a_seq_lengths_ptr = case_config.inputs[2]
        a_offsets_ptr = case_config.inputs[3]
        max_seq_len_attr = case_config.inputs[4]

        # 从 offsets 推断 B，生成各 batch 方阵边长
        B = self._resolve_B(a_offsets_ptr)
        lengths_seq = self._generate_seq_lengths(B)
        offsets = self._cumsum_squares(lengths_seq)
        total_seq = offsets[-1]
        max_val = max(lengths_seq)

        # 自洽 shape：a_ptr=[Σseq²]、b_ptr=[max, max] 方阵、lengths=[B]、offsets=[B+1]
        if hasattr(a_ptr, 'shape'):
            a_ptr.shape = [total_seq]
        if hasattr(b_ptr, 'shape'):
            b_ptr.shape = [max_val, max_val]
        if hasattr(a_seq_lengths_ptr, 'shape'):
            a_seq_lengths_ptr.shape = [B]
        if hasattr(a_offsets_ptr, 'shape'):
            a_offsets_ptr.shape = [B + 1]

        a_seq_lengths_ptr.outlier_values = lengths_seq
        a_offsets_ptr.outlier_values = offsets

        if hasattr(max_seq_len_attr, 'range_values'):
            max_seq_len_attr.range_values = [max_val]

        # a_ptr 与 b_ptr dtype 一致
        if hasattr(a_ptr, 'dtype') and hasattr(b_ptr, 'dtype'):
            b_ptr.dtype = a_ptr.dtype

        return case_config

    def _resolve_B(self, offset_tensor):
        """从offset tensor推断batch数量"""
        if hasattr(offset_tensor, 'shape') and offset_tensor.shape:
            offset_len = offset_tensor.shape[0]
            if offset_len > 1:
                return offset_len - 1
        if hasattr(offset_tensor, 'dim_values') and offset_tensor.dim_values:
            import random

            offset_len = random.choice(offset_tensor.dim_values)
            if offset_len > 1:
                return offset_len - 1
        import random

        return random.choice([2, 4, 8])

    def _hash_range(self, i, lo, hi, salt=0):
        """确定性哈希：把索引i映射到[lo, hi]内的散乱值"""
        x = ((i + 1) * 2654435761 + salt * 40503) & 0xFFFFFFFF
        x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
        x = (x ^ (x >> 16)) & 0x7FFFFFFF
        return lo + x % (hi - lo + 1)

    def _generate_seq_lengths(self, B):
        """生成 B 个方阵边长（random 增加多样性）"""
        import random

        lo, hi = 16, 128
        return [random.randint(lo, hi) for _ in range(B)]

    def _cumsum_squares(self, lengths):
        """计算 seq_len 平方的 cumsum"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length * length)
        return offsets
