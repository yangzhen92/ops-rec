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
jagged_jagged_bmm_jagged_out_kernel 算子参数约束生成器

语义（FBGEMM 官方测试 + CPU 参考）：逐 batch C_i = A_i @ B_i^T
    jagged_A [Σm_i, K]
    jagged_B [K, Σn_i]（转置布局）
    jagged_C [Σ(m_i·n_i)]         —— 输出，wrapper 内部构建，非输入
    约束：lengths_n = lengths_m，lengths_mn = lengths_m·lengths_n

设计原则（Generator 只做约束，不固定用例）：
1. shape 的多样性由 ATK 采样（B、total_m、K 都随 yaml 分布变化），Generator 不写死。
2. Generator 只保证「自洽」：三个量之间可能矛盾的地方拉平——
     - jagged_B 行数 = jagged_A 列数（共享维 K）
     - jagged_B 列数 = jagged_A 行数（因 n=m ⇒ Σn=Σm）
     - 三个 offset 都是 [B+1]
     - total_m ≥ B（保证每个 batch 至少 1 行，切分后无空 batch）
3. 每 batch 的 m_i 划分是 jagged 元信息，无法通过 tensor/outlier_values 注入
   （outlier_values 只能存 2 个数），因此由 API 用 ATK 采样到的 total_m 确定性切分。
   注意：C 的规模 Σ(m_i·n_i) 是「输出」，Generator 无需知道逐 batch 长度。
4. 全程无随机决策依赖具体数据值 —— shape 由 ATK 定，切分在 API 内确定性完成，
   GPU/NPU 输入一致（single_bm 精度对比才有效）。
"""
import random
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_jagged_jagged_bmm_jagged_out_kernel")
class JaggedJaggedBmmJaggedOutKernelGenerator(CaseGenerator):
    """jagged_jagged_bmm_jagged_out_kernel 算子参数约束生成器"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if len(case_config.inputs) < 6:
            return case_config

        jagged_A = case_config.inputs[0]
        jagged_B = case_config.inputs[1]
        offsets_m = case_config.inputs[2]
        offsets_n = case_config.inputs[3]
        offsets_mn = case_config.inputs[4]
        max_seq_len = case_config.inputs[5]

        # ATK 采样：B（offset 维决定）、total_m 与 K（jagged_A 两维）
        B = self._resolve_B(offsets_m)
        total_m = self._resolve_total_m(jagged_A)
        K = self._resolve_K(jagged_A)

        # 约束1：total_m ≥ B，保证切分后每 batch 至少 1 行（无空 batch）
        if total_m < B:
            total_m = B

        # 约束2：自洽 shape
        #   jagged_A [total_m, K]
        #   jagged_B [K, total_m]（转置布局；因 n=m ⇒ Σn=Σm=total_m）
        #   三个 offset 都是 [B+1]
        self._set_shape(jagged_A, [total_m, K])
        self._set_shape(jagged_B, [K, total_m])
        self._set_shape(offsets_m, [B + 1])
        self._set_shape(offsets_n, [B + 1])
        self._set_shape(offsets_mn, [B + 1])

        # 约束3：max_seq_len 需 ≥ 任何单 batch 长度。API 用「变长」切分，单 batch 上界
        # 无法用 ceil(total_m/B) 卡死（变长会超过均值），故取安全上界 total_m。
        # kernel 用 max_seq_len 定 grid，偏大只多空转 block、不越界；API 会自行重算为
        # max(lengths_m)，此处仅让 yaml 参数取值合理。
        if hasattr(max_seq_len, 'range_values'):
            max_seq_len.range_values = [total_m]

        # 约束4：A、B dtype 一致
        if hasattr(jagged_A, 'dtype') and hasattr(jagged_B, 'dtype'):
            jagged_B.dtype = jagged_A.dtype

        return case_config

    def _resolve_B(self, offset):
        """B = offset_len - 1（offset 长度可信）"""
        offset_len = self._get_first_dim(offset)
        if offset_len is not None and offset_len > 1:
            return offset_len - 1
        return random.choice([2, 4, 8, 16])

    def _resolve_total_m(self, tensor):
        """jagged_A 行数（ATK 采样值，可信）"""
        rows = self._get_first_dim(tensor)
        if rows is not None and rows > 0:
            return rows
        return random.choice([64, 128, 256, 512])

    def _resolve_K(self, tensor):
        """共享维 K = jagged_A 列数（ATK 采样值，可信）"""
        k = self._get_second_dim(tensor)
        if k is not None and k > 0:
            return k
        return random.choice([16, 32, 64, 128])

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