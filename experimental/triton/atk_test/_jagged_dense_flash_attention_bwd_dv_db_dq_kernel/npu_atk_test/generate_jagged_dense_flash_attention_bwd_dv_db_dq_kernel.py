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
#     limitations under the License.
# ==============================================================================
"""
_jagged_dense_flash_attention_bwd_dv_db_dq_kernel 算子参数约束生成器

参数约束说明：
1. D（hidden dimension）必须是 2 的幂次方，因为 triton arange 要求范围是 2 的幂次方
2. K: dense tensor, shape [B, D, T]
3. Q, V: jagged tensor, shape [sum_B, D]
4. Out, do: shape [B, T, D]
5. lse: shape [B, T]
6. attn_bias: shape [B, N, T]
7. jagged_offsets: shape [B + 1]，其中 jagged_offsets[-1] = sum_B
8. sum_B = Q.shape[0] = V.shape[0]
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


def is_power_of_2(n):
    """判断 n 是否是 2 的幂次方"""
    return n > 0 and (n & (n - 1)) == 0


def get_next_power_of_2(n):
    """获取大于等于 n 的最小 2 的幂次方"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@GENERATOR_REGISTRY.register("generate_jagged_dense_flash_attention_bwd_dv_db_dq_kernel")
class JaggedDenseFlashAttentionBwdDvDbDqGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """
        用例参数约束修改入口
        :param case_config: 生成的用例信息
        :return: 返回修改后符合参数间约束关系的用例
        """
        # 获取输入参数
        Q = case_config.inputs[0]
        K = case_config.inputs[1]
        V = case_config.inputs[2]
        Out = case_config.inputs[3]
        lse = case_config.inputs[4]
        do = case_config.inputs[5]
        attn_bias = case_config.inputs[6]
        jagged_offsets = case_config.inputs[7]

        # 获取各参数的 shape
        q_shape = list(Q.shape) if hasattr(Q, 'shape') else [1, 1]
        k_shape = list(K.shape) if hasattr(K, 'shape') else [1, 1, 1]
        v_shape = list(V.shape) if hasattr(V, 'shape') else [1, 1]
        out_shape = list(Out.shape) if hasattr(Out, 'shape') else [1, 1, 1]
        lse_shape = list(lse.shape) if hasattr(lse, 'shape') else [1, 1]
        do_shape = list(do.shape) if hasattr(do, 'shape') else [1, 1, 1]
        ab_shape = list(attn_bias.shape) if hasattr(attn_bias, 'shape') else [1, 1, 1]
        jo_shape = list(jagged_offsets.shape) if hasattr(jagged_offsets, 'shape') else [1]

        # ====== 约束1：D 必须是 2 的幂次方，且 D >= 16（triton dot 要求 K >= 16）======
        if len(k_shape) >= 2:
            D = k_shape[1]
            if not is_power_of_2(D) or D < 16:
                D = max(16, get_next_power_of_2(D))
                k_shape[1] = D

        # ====== 从 K 获取 B 和 T ======
        B = k_shape[0] if len(k_shape) >= 1 else 1
        T = k_shape[2] if len(k_shape) >= 3 else (out_shape[1] if len(out_shape) >= 2 else 1)

        # ====== 约束：max_seq_len >= T（避免序列被截断）======
        # 获取 max_seq_len 参数（倒数第二个输入参数）
        max_seq_len = case_config.inputs[8] if len(case_config.inputs) > 8 else None
        if max_seq_len is not None:
            # max_seq_len 是 scalar 类型，检查其 range_values
            # scalar 的 range_values 可能是一个值或一个列表
            if hasattr(max_seq_len, 'range_values') and max_seq_len.range_values is not None:
                rv = max_seq_len.range_values
                # 如果 range_values 是一个列表，取第一个元素作为最小值
                if isinstance(rv, list):
                    min_val = rv[0] if isinstance(rv[0], (int, float)) else None
                else:
                    min_val = rv if isinstance(rv, (int, float)) else None

                # 如果 max_seq_len < T，则调整为 T
                if isinstance(min_val, (int, float)) and min_val < T:
                    if isinstance(rv, list) and len(rv) > 1 and isinstance(rv[1], (int, float)):
                        max_seq_len.range_values = [T, rv[1]]
                    else:
                        max_seq_len.range_values = T

        # ====== 约束2：K 形状 [B, D, T] ======
        k_shape = [B, D, T]
        K.shape = k_shape

        # ====== 约束3：Q, V 形状 [sum_B, D] ======
        # sum_B 根据 B 和 T 估算一个合理值
        sum_B = B * max(1, T // 2)  # 假设平均序列长度约为 T/2
        sum_B = max(64, min(sum_B, 8192))  # 限制在合理范围内
        q_shape = [sum_B, D]
        v_shape = [sum_B, D]
        Q.shape = q_shape
        V.shape = v_shape

        # ====== 约束4：Out, do 形状 [B, T, D] ======
        out_shape = [B, T, D]
        do_shape = [B, T, D]
        Out.shape = out_shape
        do.shape = do_shape

        # ====== 约束5：lse 形状 [B, T] ======
        lse_shape = [B, T]
        lse.shape = lse_shape

        # ====== 约束6：attn_bias 形状 [B, N, T] ======
        # N (number of heads) 使用默认值 8
        N = ab_shape[1] if len(ab_shape) >= 2 else 8
        ab_shape = [B, N, T]
        attn_bias.shape = ab_shape

        # ====== 约束7：jagged_offsets 形状 [B + 1] ======
        jo_shape = [B + 1]
        jagged_offsets.shape = jo_shape

        return case_config
