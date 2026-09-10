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

# generate_triton_batched_dense_vec_jagged_2d_matmul.py
"""
triton_batched_dense_vec_jagged_2d_matmul 算子参数约束生成器

维度语义（关键）：
- B: batch数（= offset长度 - 1）
- H: head数（独立）
- D: embedding维度（独立）
- max_L: 最大序列长度（dense的第二维，独立）
- total_L: values的行数 = sum(lengths) = offsets[-1]

约束链：offset长度 → B；H/D/max_L 独立选；total_L 由段长cumsum得到
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_batched_dense_vec_jagged_2d_matmul")
class TritonBatchedDenseVecJagged2dMatmulGenerator(CaseGenerator):
    """triton_batched_dense_vec_jagged_2d_matmul 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 3:
            return case_config

        dense = case_config.inputs[0]
        values = case_config.inputs[1]
        offset = case_config.inputs[2]

        # 步骤1: 确定 B（从 offset.shape[0] 推导）
        B = self._resolve_B(offset)

        # 步骤2: 确定 H（head数，独立选择）
        H = self._resolve_dim(dense, default=8, prefer_second=False, divisor=B)

        # 步骤3: 确定 D（embedding维度，独立选择）
        D = self._resolve_D(values, H)

        # ====== 新增：强制 D >= 16，避免向量化退化为标量 ======
        D = max(D, 16)  # 或者 32、64 等常用值，根据硬件向量宽度选择

        # 步骤4: 确定 max_L（dense第二维，独立选择）
        max_L = self._resolve_max_L(dense)

        # 步骤5: 确定性生成每段长度 lengths
        lengths = self._generate_lengths(B, max_L, values)
        total_L = sum(lengths)

        # dense: [B*H, max_L]
        self._set_shape(dense, [B * H, max_L])
        self._set_shape(values, [total_L, H * D])
        self._set_shape(offset, [B + 1])

        # ========== 注入合法offset值（cumsum(lengths)）==========
        offset.outlier_values = self._cumsum_offsets(lengths)

        # dtype 一致性
        if hasattr(dense, 'dtype') and hasattr(values, 'dtype'):
            values.dtype = dense.dtype

        return case_config

    def _generate_lengths(self, B, max_L, values):
        lengths = []
        for i in range(B):
            # 确定性变化：在 [max_L//2, max_L] 内波动
            seg = (max_L // 2) + (i % 4) * (max_L // 2)
            seg = min(seg, max_L)  # clamp 到 max_L：消除超长死行
            seg = max(1, seg)  # 每段至少 1 行
            lengths.append(seg)
        return lengths

    def _cumsum_offsets(self, lengths):
        """由lengths生成offset：[0, l0, l0+l1, ...]"""
        offsets = [0]
        for seg in lengths:
            offsets.append(offsets[-1] + seg)
        return offsets

    def _resolve_B(self, offset):
        """解析 B = offset_len - 1"""
        offset_len = self._get_first_dim(offset)
        if offset_len is not None and offset_len > 1:
            return offset_len - 1
        return 4

    def _resolve_dim(self, dense, default, prefer_second, divisor):
        """解析 H：确保 dense.shape[0] = B * H 可整除"""
        bh = self._get_first_dim(dense)
        if bh is not None and bh > 0 and divisor > 0 and bh % divisor == 0:
            h = bh // divisor
            if h > 0:
                return h
        return default

    def _resolve_D(self, values, H):
        """解析 D，确保 values.shape[1] = H * D 可整除"""
        hd = self._get_second_dim(values)
        if hd is not None and hd > 0 and H > 0 and hd % H == 0:
            d = hd // H
            if d > 0:
                return d
        return 64

    def _resolve_max_L(self, dense):
        """解析 max_L（dense第二维）"""
        m = self._get_second_dim(dense)
        if m is not None and m > 0:
            return m
        return 32

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
