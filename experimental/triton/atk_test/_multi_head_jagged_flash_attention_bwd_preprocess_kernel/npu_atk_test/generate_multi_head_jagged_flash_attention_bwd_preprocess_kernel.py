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


@GENERATOR_REGISTRY.register("generate__multi_head_jagged_flash_attention_bwd_preprocess_kernel")
class MultiHeadJaggedFlashAttentionBwdPreprocessKernelGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 3:
            return case_config

        o = case_config.inputs[0]
        do = case_config.inputs[1]
        o_offset = case_config.inputs[2]
        num_heads_attr = case_config.inputs[3] if len(case_config.inputs) > 3 else None
        max_seq_len = case_config.inputs[4] if len(case_config.inputs) > 4 else None

        num_heads = self._get_first_dim(o)
        total_seq = self._get_second_dim(o)
        D = self._get_third_dim(o)

        if num_heads_attr is not None and hasattr(num_heads_attr, 'range_values'):
            rv = num_heads_attr.range_values
            if isinstance(rv, list) and len(rv) == 2:
                valid_heads = [h for h in [2, 4, 8, 16] if rv[0] <= h <= rv[1]]
                num_heads = valid_heads[0] if valid_heads else 4
            else:
                num_heads = int(rv) if not isinstance(rv, list) else 4
        elif num_heads is None:
            num_heads = 4

        if hasattr(do, 'shape'):
            do.shape = [num_heads, total_seq, D]

        offset_len = self._get_first_dim(o_offset)
        if offset_len is not None and offset_len > 1:
            B = offset_len - 1
        else:
            B = 8  # 默认值

        if total_seq < B:
            total_seq = B
            self._set_shape(o, [num_heads, total_seq, D])
            self._set_shape(do, [num_heads, total_seq, D])

        self._set_shape(o_offset, [B + 1])

        D = self._round_to_power_of_2(D)
        self._set_shape(o, [num_heads, total_seq, D])
        self._set_shape(do, [num_heads, total_seq, D])

        num_heads = self._round_to_power_of_2_small(num_heads)
        self._set_shape(o, [num_heads, total_seq, D])
        self._set_shape(do, [num_heads, total_seq, D])

        if max_seq_len is not None and hasattr(max_seq_len, 'range_values'):
            min_max_seq_len = max(16, (total_seq + B - 1) // B)
            max_seq_len.range_values = [min_max_seq_len]

        if hasattr(o, 'dtype') and hasattr(do, 'dtype'):
            do.dtype = o.dtype

        return case_config

    def _get_first_dim(self, tensor):
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list) and len(tensor.shape) > 0:
            val = tensor.shape[0]
            return val if isinstance(val, int) else None
        return None

    def _get_second_dim(self, tensor):
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list) and len(tensor.shape) > 1:
            val = tensor.shape[1]
            return val if isinstance(val, int) else None
        return None

    def _get_third_dim(self, tensor):
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list) and len(tensor.shape) > 2:
            val = tensor.shape[2]
            return val if isinstance(val, int) else None
        return None

    def _set_shape(self, tensor, target_shape):
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list):
            tensor.shape[:] = target_shape
        else:
            tensor.shape = target_shape

    def _round_to_power_of_2(self, value):
        if value <= 16:
            return 16
        elif value <= 32:
            return 32
        elif value <= 64:
            return 64
        elif value <= 128:
            return 128
        elif value <= 256:
            return 256
        else:
            return 512

    def _round_to_power_of_2_small(self, value):
        if value <= 2:
            return 2
        elif value <= 4:
            return 4
        elif value <= 8:
            return 8
        elif value <= 16:
            return 16
        else:
            return 32
