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
# generate_split_2D_jagged_w_prefix.py
"""split_2D_jagged_w_prefix kernel 的 ATK shape 约束生成器。

计算过程分析
=============
split_2D_jagged_w_prefix 是一个纯数据搬运算子（无 reduce/bmm/element-wise 运算），
核心操作为 tl.load → tl.store，从 JaggedIn 按地址映射规则拆分写入 OutA 和 OutB。

地址映射逻辑（jagged+jagged 模式, IS_DENSE_A=False, IS_DENSE_B=False, IS_REPLACE=False）:
  1. 读 OffsetsA[z], OffsetsA[z+1] → seq_start_a, seq_len_a
  2. 读 OffsetsB[z], OffsetsB[z+1] → seq_start_b, seq_len_b
  3. 每 batch 输出总长度 seq_len = seq_len_a + seq_len_b
  4. 输入偏移: seq_start = seq_start_a + seq_start_b
  5. 输出路由:
     - n_prefix_to_B <= off_n < seq_len_a + n_prefix_to_B → OutA
     - 否则 → OutB

kernel grid: (max_seq_len, B)
其中 max_seq_len = max(seq_len_a + seq_len_b) over batches
B = OffsetsA.shape[0] - 1 = OffsetsB.shape[0] - 1

重要特性:
  - 无数学运算：不存在精度累积问题
  - 无 ALLOW_TF32 依赖：不使用 tl.dot()
  - dtype 直通：输入什么 dtype 输出就是什么 dtype
  - 测试模式：仅测试 jagged+jagged (IS_DENSE_A=False, IS_DENSE_B=False, IS_REPLACE=False)
                带 n_prefix_to_B=0（默认 split 模式）
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


# D 维候选：覆盖 BLOCK_D = next_power_of_2(D) 的边界
_D_CANDIDATES = (1, 2, 15, 16, 31, 32, 63, 64, 65, 128, 256, 512, 1024, 2048, 4096, 8192)
# batch 候选
_B_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
# 总元素数上限: JaggedIn(L*D) + OutA(L_A*D) + OutB(L_B*D) ≈ 2*L*D (L = L_A + L_B)
_MAX_TOTAL_ELEMENTS = 1 << 24
# 单序列长度上限
_MAX_SEQ_LEN = 65536
# coreDim 上限（二维 grid: max_seq_len × B）
_MAX_CORE_DIM = 65535


def _nearest_supported_d(value: int) -> int:
    """将 YAML 采样维度映射到 D 候选集。"""
    return min(_D_CANDIDATES, key=lambda candidate: (abs(candidate - value), candidate))


def _select_batch_size(offset_length: int, requested_d: int) -> int:
    """按确定性规则选择 batch 大小。"""
    base_index = max(1, offset_length - 1)
    seed = (base_index + requested_d) % len(_B_CANDIDATES)
    return _B_CANDIDATES[seed]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@GENERATOR_REGISTRY.register("generate_triton_split_2D_jagged_w_prefix")
class Split2DJaggedWPrefixGenerator(CaseGenerator):
    """将 YAML 采样 shape 修正为 kernel 可执行的 B、L_A、L_B、D 组合。

    kernel grid: (max_seq_len, B)，二维 grid 受 NPU coreDim 上限约束。
    """

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        jagged_in, offsets_a, offsets_b = case_config.inputs[:3]

        requested_l = int(jagged_in.shape[0])
        requested_d = int(jagged_in.shape[1])
        requested_offset_length = int(offsets_a.shape[0])

        d = _nearest_supported_d(max(1, requested_d))
        batch_size = _select_batch_size(requested_offset_length, requested_d)

        # JaggedIn 的 D 维修正
        jagged_in.shape[1] = d

        # Batch 数: B = offsets.shape[0] - 1，保证 B >= 1
        batch_size = max(1, batch_size)
        offsets_a.shape = [batch_size + 1]
        offsets_b.shape = [batch_size + 1]

        # L = L_A + L_B，且 L >= B（每个 batch 至少一个 token）
        # 总内存: JaggedIn(L*D) + OutA(≈L/2*D) + OutB(≈L/2*D) ≈ 2*L*D <= MAX
        max_l_by_memory = max(batch_size, _MAX_TOTAL_ELEMENTS // (2 * d))
        total_length = min(max(batch_size, requested_l), _MAX_SEQ_LEN, max_l_by_memory)

        # 检查 coreDim 约束（近似：每 batch 约 l/batch_size 个 token）
        approx_max_seq_len = _cdiv(total_length, batch_size)
        core_dim = approx_max_seq_len * batch_size
        while core_dim > _MAX_CORE_DIM and total_length > batch_size:
            total_length = total_length * 3 // 4
            approx_max_seq_len = _cdiv(total_length, batch_size)
            core_dim = approx_max_seq_len * batch_size

        jagged_in.shape[0] = total_length
        offsets_a.shape = [batch_size + 1]
        offsets_b.shape = [batch_size + 1]

        return case_config
