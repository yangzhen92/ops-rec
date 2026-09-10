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
jagged_flash_attention_bwd_preprocess_basic_kernel 算子参数约束生成器

语义：Flash Attention 反向传播预处理，计算 delta = sum(o * do, axis=1)
    o [total_seq, D]           前向输出
    do [total_seq, D]          输出梯度
    o_offset [B+1]             jagged offset
    delta [total_seq]          输出: o 和 do 的逐元素乘积后按行求和

约束：
    1. o.shape == do.shape
    2. o_offset[-1] <= total_seq
    3. D 通常是 2 的幂次方（16, 32, 64, 128）
    4. max_seq_len >= max(seq_lengths)
"""
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate__jagged_flash_attention_bwd_preprocess_basic_kernel")
class JaggedFlashAttentionBwdPreprocessBasicKernelGenerator(CaseGenerator):
    """jagged_flash_attention_bwd_preprocess_basic_kernel 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 3:
            return case_config

        o = case_config.inputs[0]
        do = case_config.inputs[1]
        o_offset = case_config.inputs[2]
        max_seq_len = case_config.inputs[3] if len(case_config.inputs) > 3 else None

        # 获取 total_seq 和 D
        total_seq = self._get_first_dim(o)
        D = self._get_second_dim(o)

        # 约束1: o.shape == do.shape
        if hasattr(do, 'shape'):
            do.shape = [total_seq, D]

        # 约束2: o_offset 的最后一个值应该 <= total_seq
        # offset 的长度决定了 batch 数
        offset_len = self._get_first_dim(o_offset)
        if offset_len is not None and offset_len > 1:
            B = offset_len - 1
        else:
            B = 8  # 默认值

        # 确保 total_seq >= B (至少每个batch有1个元素)
        if total_seq < B:
            total_seq = B
            self._set_shape(o, [total_seq, D])
            self._set_shape(do, [total_seq, D])

        # 更新 o_offset shape
        self._set_shape(o_offset, [B + 1])

        # 约束3: D 通常是 2 的幂次方
        D = self._round_to_power_of_2(D)
        self._set_shape(o, [total_seq, D])
        self._set_shape(do, [total_seq, D])

        # 约束4: max_seq_len 应该 >= 实际的最大序列长度
        # 保守估计：max_seq_len >= total_seq / B
        if max_seq_len is not None and hasattr(max_seq_len, 'range_values'):
            min_max_seq_len = max(16, (total_seq + B - 1) // B)
            max_seq_len.range_values = [min_max_seq_len]

        # 约束5: o 和 do 的 dtype 必须一致
        if hasattr(o, 'dtype') and hasattr(do, 'dtype'):
            do.dtype = o.dtype

        return case_config

    def _get_first_dim(self, tensor):
        """安全获取 tensor.shape[0]"""
        if (hasattr(tensor, 'shape') and isinstance(tensor.shape, list)
                and len(tensor.shape) > 0):
            val = tensor.shape[0]
            return val if isinstance(val, int) else None
        return None

    def _get_second_dim(self, tensor):
        """安全获取 tensor.shape[1]"""
        if (hasattr(tensor, 'shape') and isinstance(tensor.shape, list)
                and len(tensor.shape) > 1):
            val = tensor.shape[1]
            return val if isinstance(val, int) else None
        return None

    def _set_shape(self, tensor, target_shape):
        """安全设置 tensor.shape"""
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list):
            tensor.shape[:] = target_shape
        else:
            tensor.shape = target_shape

    def _round_to_power_of_2(self, value):
        """将值向上取整到最近的 2 的幂次方"""
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