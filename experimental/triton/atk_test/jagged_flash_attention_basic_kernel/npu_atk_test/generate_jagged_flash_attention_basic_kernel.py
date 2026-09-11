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
"""jagged_flash_attention_basic_kernel 算子参数约束生成器"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_jagged_flash_attention_basic_kernel")
class JaggedFlashAttentionBasicKernelGenerator(CaseGenerator):
    MAX_TOTAL_SEQ = 256
    MIN_TOTAL_SEQ = 32
    MAX_D = 64
    MIN_D = 16
    MAX_B = 8
    MIN_B = 1

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 5:
            return case_config
        q = case_config.inputs[0]
        k = case_config.inputs[1]
        v = case_config.inputs[2]
        offset = case_config.inputs[3]
        max_seq_len_attr = case_config.inputs[4]
        total_seq = self._get_first_dim(q)
        D = self._get_second_dim(q)
        if total_seq is None or D is None:
            return case_config
        D = self._round_to_power_of_2(D)
        total_seq = max(self.MIN_TOTAL_SEQ, min(total_seq, self.MAX_TOTAL_SEQ))
        D = max(self.MIN_D, min(D, self.MAX_D))
        offset_len = self._get_first_dim(offset)
        if offset_len is not None and offset_len > 1:
            B = offset_len - 1
        else:
            B = 8
        B = max(self.MIN_B, min(B, self.MAX_B))
        total_seq = max(total_seq, B)
        offsets = self._gen_offsets(B, total_seq)
        self._set_shape(q, [total_seq, D])
        self._set_shape(k, [D, total_seq])
        self._set_shape(v, [total_seq, D])
        self._set_shape(offset, [B + 1])
        offset.outlier_values = offsets
        avg_seq = (total_seq + B - 1) // B
        max_seq_len = max(16, self._next_power_of_2(avg_seq))
        if hasattr(max_seq_len_attr, 'range_values'):
            max_seq_len_attr.range_values = [max_seq_len]
        if hasattr(q, 'dtype'):
            if hasattr(k, 'dtype'):
                k.dtype = q.dtype
            if hasattr(v, 'dtype'):
                v.dtype = q.dtype
        return case_config

    def _gen_offsets(self, B, total_seq):
        if B <= 0:
            return [0, total_seq]
        base = total_seq // B
        rem = total_seq % B
        offsets = [0]
        cur = 0
        for i in range(B):
            seg = base + (1 if i < rem else 0)
            seg = max(1, seg)
            cur += seg
            cur = min(cur, total_seq)
            offsets.append(cur)
        offsets[-1] = total_seq
        return offsets

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
        else:
            return 256

    def _next_power_of_2(self, n):
        p = 1
        while p < n:
            p *= 2
        return p
