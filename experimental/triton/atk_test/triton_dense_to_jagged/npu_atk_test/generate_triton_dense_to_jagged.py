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
dense_to_jagged 算子参数约束生成器

本文件用于在 ATK 用例生成时，对参数进行约束检查和修正。
确保生成的用例满足算子的限制条件。

限制条件（根据 limit.txt）：
1. dense 是 3 维或 4 维，最后一维 D（特征维度）<= 8192
2. 3D dense [B, seq, D]: B < seq < D < 8192
   - jagged_offsets 长度 1，shape [B+1]
   - jagged_offsets[0][0] = 0（偏移从 0 开始）
   - 2*B <= jagged_offsets[0][-1] <= B*seq（最后一个偏移）
   - 相邻差值不超过 seq（每个 batch 元素的长度不超过 seq）
3. 4D dense [B1, B2, seq, D]: B1 < B2 < seq < D < 8192
   - jagged_offsets 长度 2，shape [B1+1] 和 [B1*B2+1]
   - jagged_offsets[i][0] = 0（偏移从 0 开始）
   - jagged_offsets[0][-1] = B1*B2（第一个 offset 的最后一个值等于总元素数）
   - jagged_offsets[0] 相邻差值不超过 B2
   - 2*B1*B2 <= jagged_offsets[1][-1] <= B1*B2*seq
   - jagged_offsets[1] 相邻差值不超过 seq
4. operation_jagged_value 的数据类型（fp32/fp16/bf16）与 dense 一致
"""
import random
import torch
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_triton_dense_to_jagged")
class DenseToJaggedGenerator(CaseGenerator):
    """dense_to_jagged 算子参数约束生成器

    在用例生成后，对参数进行约束检查和修正。
    """

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """
        用例参数约束修改入口

        ATK 工具在生成用例后会调用这个函数，
        可以在此处修改用例参数以满足约束条件。

        参数:
            case_config: 生成的用例信息，包含所有输入参数

        返回:
            修改后的用例信息
        """
        # 获取输入参数（按 yaml 中定义的顺序）
        dense = case_config.inputs[0]
        jagged_offsets_input = case_config.inputs[1]
        operation_function = case_config.inputs[2] if len(case_config.inputs) > 2 else None
        operation_jagged_value = case_config.inputs[3] if len(case_config.inputs) > 3 else None

        # 获取 dense tensor 的形状信息
        dense_shape = list(dense.shape) if hasattr(dense, 'shape') else [1, 1, 1]
        dense_dim = len(dense_shape)  # 维度数量（3 或 4）
        D = dense_shape[-1]

        # ==================== 约束1：特征维度 D <= 8192 ====================
        # triton 的 arange 要求 D 是 2 的幂次方，且有上限
        if D > 8192:
            D = 8192
            dense_shape[-1] = D
            dense.shape = dense_shape

        # ==================== 情况1: 3D dense [B, seq, D] ====================
        # 例如：[2, 4, 8] 表示 B=2, seq=4, D=8
        # 约束：B < seq < D，且 B <= D/4
        if dense_dim == 3:
            B = dense_shape[0]    # 批次大小
            seq = dense_shape[1]  # 序列长度

            # 确保 B <= D/4（留出足够空间给 seq）
            if B > D // 4:
                B = max(2, D // 4)
                dense_shape[0] = B

            # 确保 B < seq < D
            if B >= seq or seq >= D:
                seq = (B + D) // 2  # 保证 B < seq < D
            dense_shape[1] = seq
            dense.shape = dense_shape

            # 更新 jagged_offsets 的约束
            # 3D dense 对应 1 个 offset，shape 为 [B+1]
            needed_offsets_count = 1
            if jagged_offsets_input is not None and hasattr(jagged_offsets_input, '__len__'):
                # 截断多余的 offset（如果 yaml 生成了多个）
                jagged_offsets_input[:] = jagged_offsets_input[:needed_offsets_count]

                if len(jagged_offsets_input) > 0:
                    offset_tensor = jagged_offsets_input[0]
                    if hasattr(offset_tensor, 'shape'):
                        offset_tensor.shape = [B + 1]  # shape 必须为 [B+1]

        # ==================== 情况2: 4D dense [B1, B2, seq, D] ====================
        # 例如：[2, 3, 4, 8] 表示 B1=2, B2=3, seq=4, D=8
        # 约束：B1 < B2 < seq < D，且 B1 <= D/4
        elif dense_dim == 4:
            B1 = dense_shape[0]   # 第一维批次大小
            B2 = dense_shape[1]   # 第二维批次大小
            seq = dense_shape[2]  # 序列长度

            # 确保 B1 <= D/4（留出足够空间给 seq）
            if B1 > D // 4:
                B1 = max(2, D // 4)
                dense_shape[0] = B1

            # 确保 B1 < B2 < seq < D
            # 如果 B1 >= B2，调整 B2
            if B1 >= B2:
                B2 = B1 + 1
                dense_shape[1] = B2

            # 如果 B2 >= seq 或 seq >= D，调整 seq（保证 B2 < seq < D）
            if B2 >= seq or seq >= D:
                seq = (B2 + D) // 2
            dense_shape[2] = seq
            dense.shape = dense_shape

            # 更新 jagged_offsets 的约束
            # 4D dense 对应 2 个 offset
            # jagged_offsets[0] shape 为 [B1+1]
            # jagged_offsets[1] shape 为 [B1*B2+1]
            needed_offsets_count = 2
            if jagged_offsets_input is not None and hasattr(jagged_offsets_input, '__len__'):
                # 截断多余的 offset
                jagged_offsets_input[:] = jagged_offsets_input[:needed_offsets_count]

                if len(jagged_offsets_input) > 0:
                    offset0_tensor = jagged_offsets_input[0]
                    if hasattr(offset0_tensor, 'shape'):
                        offset0_tensor.shape = [B1 + 1]

                if len(jagged_offsets_input) > 1:
                    offset1_tensor = jagged_offsets_input[1]
                    if hasattr(offset1_tensor, 'shape'):
                        offset1_tensor.shape = [B1 * B2 + 1]

        # ==================== 约束2：jagged_offsets 数据类型一致 ====================
        # 所有 offset tensor 必须使用相同的数据类型（int32 或 int64）
        dtypes = set()
        for offset_tensor in jagged_offsets_input:
            if offset_tensor is not None and hasattr(offset_tensor, 'dtype'):
                dtypes.add(offset_tensor.dtype)

        if len(dtypes) > 1:
            # 统一为 int64
            unified_dtype = 'int64'
            for offset_tensor in jagged_offsets_input:
                if offset_tensor is not None and hasattr(offset_tensor, 'dtype'):
                    offset_tensor.dtype = unified_dtype

        # ==================== 约束4：jagged_offsets 值约束 ====================
        # 3D: 2*B <= jagged_offsets[0][-1] <= B*seq，相邻差值不超过 seq
        # 4D: jagged_offsets[0][-1] == B1*B2，相邻差值不超过 B2
        #      2*B1*B2 <= jagged_offsets[1][-1] <= B1*B2*seq，相邻差值不超过 seq
        #      所有 offset 必须递增（jagged_offsets[i][j+1] >= jagged_offsets[i][j]）
        #      所有 offset 必须从0开始（jagged_offsets[i][0] == 0）
        for idx, offset_tensor in enumerate(jagged_offsets_input):
            if offset_tensor is None or not hasattr(offset_tensor, 'value'):
                continue

            values = offset_tensor.value if hasattr(offset_tensor, 'value') else []
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                continue

            # 确保从0开始
            if values[0] != 0:
                values[0] = 0

            # 确保递增，并限制相邻差值
            if dense_dim == 3:
                max_diff = seq
            else:  # 4D
                if idx == 0:
                    max_diff = B2
                else:
                    max_diff = seq

            for i in range(1, len(values)):
                if values[i] < values[i-1]:
                    values[i] = values[i-1]
                if values[i] - values[i-1] > max_diff:
                    values[i] = values[i-1] + max_diff

            # 限制最后一个值
            if dense_dim == 3:
                # 2*B <= jagged_offsets[0][-1] <= B*seq
                min_last = 2 * B
                max_last = B * seq
                if values[-1] < min_last:
                    values[-1] = min_last
                elif values[-1] > max_last:
                    values[-1] = max_last
            else:  # 4D
                if idx == 0:
                    # jagged_offsets[0][-1] == B1*B2
                    values[-1] = B1 * B2
                else:
                    # 2*B1*B2 <= jagged_offsets[1][-1] <= B1*B2*seq
                    min_last = 2 * B1 * B2
                    max_last = B1 * B2 * seq
                    if values[-1] < min_last:
                        values[-1] = min_last
                    elif values[-1] > max_last:
                        values[-1] = max_last

            offset_tensor.value = values

        # ==================== 约束3：operation_jagged_value 的数据类型 ====================
        # 当 operation_function 不为 null 时，operation_jagged_value 的数据类型需与 dense 一致
        op_func_val = None
        if operation_function is not None and hasattr(operation_function, 'range_values') and operation_function.range_values:
            op_func_val = operation_function.range_values[0]

        if op_func_val is None or str(op_func_val) == 'null' or str(op_func_val) == 'None':
            # operation_function 为 null 时，不需要处理 operation_jagged_value
            pass
        else:
            # operation_function 为 "add" 或 "mul" 时，operation_jagged_value 的数据类型必须与 dense 相同
            dense_dtype = None
            if hasattr(dense, 'dtype'):
                dense_dtype = dense.dtype
            if operation_jagged_value is not None and dense_dtype is not None:
                operation_jagged_value.dtype = dense_dtype

        return case_config