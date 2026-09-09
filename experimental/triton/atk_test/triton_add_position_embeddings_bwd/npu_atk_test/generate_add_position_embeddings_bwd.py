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

# generate_add_position_embeddings_bwd.py
"""
_add_position_embeddings_bwd_kernel 算子参数约束生成器

约束关系：
1. Jagged.shape[0] == sum(seq_offsets段长度) == seq_offsets[-1]
2. Jagged.shape[1] == DenseOut.shape[1]  # D维度对齐
3. DenseOut.shape[0] == len(high_inds) == len(seq_offsets) - 1 == B
4. high_inds[i] < seq_offsets[i+1] - seq_offsets[i] (每段的高位索引不能超过段长)
5. Jagged.dtype == DenseOut.dtype
6. seq_offsets单调递增, 首0尾sum
7. JaggedOut.shape == Jagged.shape (当SCALE_JAGGED时)
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_add_position_embeddings_bwd")
class AddPositionEmbeddingsBwdGenerator(CaseGenerator):
    """_add_position_embeddings_bwd_kernel 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """用例参数约束修改入口"""
        Jagged = case_config.inputs[0]
        seq_offsets = case_config.inputs[1]
        high_inds = case_config.inputs[2]
        DenseOut = case_config.inputs[3]

        # === 1. dtype一致性 ===
        # Jagged和DenseOut的dtype必须一致
        if hasattr(Jagged, 'dtype') and hasattr(DenseOut, 'dtype'):
            DenseOut.dtype = Jagged.dtype

        # === 2. D维度对齐 ===
        # Jagged.shape[1] == DenseOut.shape[1]
        if hasattr(Jagged, 'shape') and hasattr(DenseOut, 'shape'):
            if len(Jagged.shape) >= 2 and len(DenseOut.shape) >= 2:
                DenseOut.shape[1] = Jagged.shape[1]

        # === 3. B维度对齐 ===
        # high_inds.shape[0] == seq_offsets.shape[0] - 1 == DenseOut.shape[0]
        if hasattr(seq_offsets, 'shape') and len(seq_offsets.shape) > 0:
            B = seq_offsets.shape[0] - 1
            if B < 1:
                B = 2
                seq_offsets.shape = [B + 1]
            if hasattr(high_inds, 'shape'):
                high_inds.shape = [B]
            if hasattr(DenseOut, 'shape'):
                DenseOut.shape[0] = B

        # === 4. Jagged总长度约束 ===
        # Jagged.shape[0] 应为 seq_offsets[-1]
        # 这里不做硬约束（因为seq_offsets值在API里才生成）
        # 确保shape兼容

        return case_config
