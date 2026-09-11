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

# generate_dense_jagged_cat_jagged_out_kernel.py
"""
dense_jagged_cat_jagged_out_kernel 算子的 ATK 约束生成器

设计原则：generator 只做「参数自洽约束」，不生成任何确定性数值。
  - 随机源：完全来自框架按 YAML 采样得到的 shape
      * B        由 a_ptr（dense [B]）的 shape[0] 决定
      * total_seq 由 b_ptr（jagged [sum(seq_len)]）的 shape[0] 决定
  - 约束：本文件只保证张量之间 shape / dtype 自洽（a=[B], b_offsets=[B+1], b/a 同 dtype）
  - 数值：offsets 的具体取值、每段长度、max_seq_len 均在运行期由 api 依据 shape 推导，
          推导是确定性的（无 random），因此同一 case 在 GPU / NPU 上结果一致。

语义：dense [B] 与 jagged [sum(seq_len)] 拼接，约束 a_ptr 与 b_ptr 必须同 dtype。
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_dense_jagged_cat_jagged_out_kernel")
class DenseJaggedCatJaggedOutGenerator(CaseGenerator):
    """dense_jagged_cat_jagged_out_kernel 约束生成器（只约束 shape/dtype，不注入数值）"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 4:
            return case_config

        a_ptr = case_config.inputs[0]
        b_ptr = case_config.inputs[1]
        b_offsets_ptr = case_config.inputs[2]

        # B 以 a_ptr（dense [B]）的 shape 为准，由框架按 YAML 随机采样
        B = self._first_dim(a_ptr)
        if B is None or B < 1:
            return case_config

        # 约束 1：a_ptr = [B]，b_offsets_ptr = [B+1]
        if hasattr(a_ptr, "shape"):
            a_ptr.shape = [B]
        if hasattr(b_offsets_ptr, "shape"):
            b_offsets_ptr.shape = [B + 1]

        # 约束 2：b_ptr = [total_seq]，total_seq 沿用框架采样值（jagged 各段长度之和）
        # 段长允许为 0，故只需保证 total_seq 为非负整数即可；给一个不低于 B 的兜底。
        total_seq = self._first_dim(b_ptr)
        if total_seq is None or total_seq < B:
            total_seq = max(B, total_seq or B)
        if hasattr(b_ptr, "shape"):
            b_ptr.shape = [total_seq]

        # 约束 3：dtype 对齐（b 必须与 a 同 dtype）
        if hasattr(a_ptr, "dtype") and hasattr(b_ptr, "dtype"):
            b_ptr.dtype = a_ptr.dtype

        return case_config

    def _first_dim(self, tensor):
        """安全读取框架已填好的 shape[0]"""
        if hasattr(tensor, "shape") and tensor.shape:
            return tensor.shape[0]
        return None
