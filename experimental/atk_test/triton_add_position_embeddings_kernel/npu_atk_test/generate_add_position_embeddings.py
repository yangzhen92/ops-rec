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
add_position_embeddings 算子 ATK 约束生成器

处理参数间的 shape 依赖关系：
1. seq_offsets, high_inds, ind_offsets 的 batch 维度 B 必须一致
2. Jagged.shape[0] 必须等于 seq_offsets[-1]
3. Dense.shape[1] 必须等于 Jagged.shape[1]（D 维度一致）
"""
from typing import Optional

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator


@GENERATOR_REGISTRY.register("generate_add_position_embeddings")
class AddPositionEmbeddingsGenerator(CaseGenerator):
    """add_position_embeddings 算子的 Generator"""

    def after_case_config(self, case_config):
        """
        调整 case_config 中的参数 shape 以满足算子约束。

        Shape 约束:
        - seq_offsets.shape[0] = B + 1
        - high_inds.shape[0] = B
        - Jagged.shape[0] = seq_offsets[-1]  (总元素数 = 最后一个 offset)
        - Jagged.shape[1] == Dense.shape[1] == D
        - ind_offsets.shape[0] = B（存在时）
        """
        jagged_input = self._get_input(case_config, "Jagged")
        seq_offsets_input = self._get_input(case_config, "seq_offsets")
        high_inds_input = self._get_input(case_config, "high_inds")
        dense_input = self._get_input(case_config, "Dense")
        ind_offsets_input = self._get_input(case_config, "ind_offsets")

        if jagged_input is None or seq_offsets_input is None or high_inds_input is None or dense_input is None:
            return case_config

        jagged_shape = getattr(jagged_input, "shape", None)
        seq_offsets_shape = getattr(seq_offsets_input, "shape", None)
        high_inds_shape = getattr(high_inds_input, "shape", None)
        dense_shape = getattr(dense_input, "shape", None)

        if not jagged_shape or not seq_offsets_shape or not high_inds_shape or not dense_shape:
            return case_config

        # ---- Step 1: 统一 D 维度 ----
        # Jagged.shape[1] 和 Dense.shape[1] 必须相等
        d_val = None
        if len(jagged_shape) >= 2:
            d_val = jagged_shape[1]
        if d_val is not None and len(dense_shape) >= 2 and dense_shape[1] != d_val:
            dense_input.shape[1] = d_val

        # ---- Step 2: 统一 batch 维度 B ----
        # seq_offsets.shape[0] = B + 1, high_inds.shape[0] = B
        b_from_seq = seq_offsets_shape[0] - 1 if len(seq_offsets_shape) >= 1 else 0
        b_from_high = high_inds_shape[0] if len(high_inds_shape) >= 1 else 0

        if b_from_seq != b_from_high:
            # 优先使用 seq_offsets 推断的 B
            b_size = b_from_seq
            high_inds_input.shape = [b_size]
        else:
            b_size = b_from_seq

        # ---- Step 2.5: 约束 high_inds[i] <= min(seq_lens[i], K-1) ----
        # Dense 只有 K 行（索引 0~K-1），high_inds[i] 超出 K-1 会导致 dense_ptrs 越界
        if len(dense_shape) >= 1:
            K = dense_shape[0]
            high_inds_range = getattr(high_inds_input, "range_values", None)
            if high_inds_range is not None and len(high_inds_range) >= 2:
                high_inds_range[1] = min(high_inds_range[1], K - 1)

        # ---- Step 3: 确保 ind_offsets 的 shape 与 B 一致 ----
        if ind_offsets_input is not None:
            ind_shape = getattr(ind_offsets_input, "shape", None)
            if ind_shape and len(ind_shape) >= 1 and ind_shape[0] != b_size:
                ind_offsets_input.shape = [b_size]

        return case_config

    @staticmethod
    def _get_input(case_config, name: str) -> Optional[object]:
        """从 case_config 中按名称获取输入。"""
        if not hasattr(case_config, "inputs") or not case_config.inputs:
            return None
        for inp in case_config.inputs:
            if getattr(inp, "name", None) == name:
                return inp
        return None