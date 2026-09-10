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
multi_head_jagged_flash_attention_fwd_kernel 算子参数约束生成器

语义：多头 Jagged Flash Attention 前向计算
    q, k, v [num_heads, total_seq, D]   多头 Query/Key/Value
    offset  [B+1]                       jagged offset
    o [num_heads, total_seq, D]         输出
    lse [num_heads, total_seq]          log-sum-exp

约束：
    1. q/k/v 同 shape [num_heads, total_seq, D]
    2. q/k/v dtype 一致
    3. D 通常是 2 的幂次方
    4. offset[-1] <= total_seq
"""
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate__multi_head_jagged_flash_attention_fwd_kernel")
class MultiHeadJaggedFlashAttentionFwdKernelGenerator(CaseGenerator):
    """multi_head_jagged_flash_attention_fwd_kernel 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 4:
            return case_config

        q = case_config.inputs[0]
        k = case_config.inputs[1]
        v = case_config.inputs[2]
        offset = case_config.inputs[3]

        # q [num_heads, total_seq, D]
        num_heads = self._get_dim(q, 0)
        total_seq = self._get_dim(q, 1)
        D = self._get_dim(q, 2)

        if num_heads is None or total_seq is None or D is None:
            return case_config

        # 约束3: D 取 2 的幂次方
        D = self._round_to_power_of_2(D)

        # 约束4: offset 长度决定 batch 数
        offset_len = self._get_dim(offset, 0)
        if offset_len is not None and offset_len > 1:
            B = offset_len - 1
        else:
            B = 8

        # 确保 total_seq >= B
        if total_seq < B:
            total_seq = B

        # 约束1: q/k/v 同 shape
        for t in (q, k, v):
            self._set_shape(t, [num_heads, total_seq, D])

        self._set_shape(offset, [B + 1])

        # 约束2: dtype 一致
        if hasattr(q, 'dtype'):
            for t in (k, v):
                if hasattr(t, 'dtype'):
                    t.dtype = q.dtype

        return case_config

    def _get_dim(self, tensor, idx):
        if (hasattr(tensor, 'shape') and isinstance(tensor.shape, list)
                and len(tensor.shape) > idx):
            val = tensor.shape[idx]
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