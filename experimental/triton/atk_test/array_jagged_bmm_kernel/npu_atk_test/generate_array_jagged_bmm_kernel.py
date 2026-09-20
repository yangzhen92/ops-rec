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
array_jagged_bmm_kernel 算子参数约束生成器

作用：
1. 按 (total_seq, B) 确定性变长切分，设置 array_A=[Σseq²]、jagged_B=[total_seq, D]、三个 offset=[B+1]
2. 约束 total_seq ≥ B、max_seq_len = max(seq)、array_A/jagged_B dtype 一致
"""

import random
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


def _split_jagged(total: int, parts: int) -> list:
    """把 total 确定性变长切成 parts 份（与 _api.py 中实现逐字一致）。"""
    if parts <= 0:
        return []
    if total <= parts:
        return [1] * parts
    seed = (total * 1000003 + parts * 9176 + 12345) & 0x7FFFFFFF
    weights = []
    for _ in range(parts):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        weights.append((seed % 997) + 1)
    wsum = sum(weights)
    lengths = [1] * parts
    remaining = total - parts
    acc = 0
    for i in range(parts):
        add = remaining * weights[i] // wsum
        lengths[i] += add
        acc += add
    leftover = remaining - acc
    i = 0
    while leftover > 0:
        lengths[i % parts] += 1
        leftover -= 1
        i += 1
    return lengths


@GENERATOR_REGISTRY.register("generate_array_jagged_bmm_kernel")
class ArrayJaggedBmmKernelGenerator(CaseGenerator):
    """array_jagged_bmm_kernel 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 6:
            return case_config
        array_A = case_config.inputs[0]
        jagged_B = case_config.inputs[1]
        offsets_am = case_config.inputs[2]
        offsets_bk = case_config.inputs[3]
        offsets_cm = case_config.inputs[4]
        max_seq_len = case_config.inputs[5]
        B = self._resolve_B(offsets_am)
        total_seq = self._resolve_total_seq(jagged_B)
        D = self._resolve_D(jagged_B)
        total_seq = max(total_seq, B)
        lengths_seq = _split_jagged(total_seq, B)
        lengths_am = [s * s for s in lengths_seq]
        total_am = sum(lengths_am)
        total_seq = sum(lengths_seq)
        self._set_shape(array_A, [total_am])
        self._set_shape(jagged_B, [total_seq, D])
        self._set_shape(offsets_am, [B + 1])
        self._set_shape(offsets_bk, [B + 1])
        self._set_shape(offsets_cm, [B + 1])
        max_val = max(lengths_seq)
        if hasattr(max_seq_len, 'range_values'):
            max_seq_len.range_values = [max_val]
        if hasattr(array_A, 'dtype') and hasattr(jagged_B, 'dtype'):
            jagged_B.dtype = array_A.dtype
        return case_config

    def _resolve_B(self, offset):
        """B = offset_len - 1（offset 长度可信）"""
        offset_len = self._get_first_dim(offset)
        if offset_len is not None and offset_len > 1:
            return offset_len - 1
        return random.choice([8, 16, 32])

    def _resolve_total_seq(self, tensor):
        """total_seq = jagged_B 第 0 维（ATK 采样值，可信；切分的总量）"""
        rows = self._get_first_dim(tensor)
        if rows is not None and rows > 0:
            return rows
        return random.choice([256, 512, 768])

    def _resolve_D(self, tensor):
        """D = jagged_B 第 1 维（ATK 采样值，可信）"""
        d = self._get_second_dim(tensor)
        if d is not None and d > 0:
            return d
        return random.choice([512, 1024, 2048])

    def _get_first_dim(self, tensor):
        """安全获取 tensor.shape[0]"""
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list) and len(tensor.shape) > 0:
            val = tensor.shape[0]
            return val if isinstance(val, int) else None
        return None

    def _get_second_dim(self, tensor):
        """安全获取 tensor.shape[1]"""
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list) and len(tensor.shape) > 1:
            val = tensor.shape[1]
            return val if isinstance(val, int) else None
        return None

    def _set_shape(self, tensor, target_shape):
        """安全设置 tensor.shape"""
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list):
            tensor.shape[:] = target_shape
        else:
            tensor.shape = target_shape
