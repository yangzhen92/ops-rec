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

"""_add_timestamp_position_embeddings_kernel 的 ATK shape 约束生成器。

关键约束：
  1. SeqEmb 为 2D [jagged_size, D]，jagged_size >= B（num_batches），
     保证每批至少一个 token。
  2. 每序列长度 <= 4096（确定性前提：query_time - ts = (seq_len-1-n)^2
     需 <= 2^24 在 fp32 中精确表示），故 jagged_size <= B * 4096。
  3. D 由 API 通过 BLOCK_D = next_power_of_2(D)（D<64 时）编译，生成器不做截断。

该生成器只调整 shape，真正的 Offsets/Lengths/TS/PosEmb/TsEmb 构造与 Out 计算
在 API 的 init_by_input_data 中执行（与教程 _layer_norm_bwd_dx 的 Mean/Rstd 推导一致）。
"""

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_add_timestamp_position_embeddings_kernel")
class AddTimestampPositionEmbeddingsKernelGenerator(CaseGenerator):
    """只调整 shape，使每个生成的配置都能被 API 安全执行。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        in_input = case_config.inputs[0]
        num_batches_input = case_config.inputs[1] if len(case_config.inputs) > 1 else None

        # ── 1. 解析 num_batches（确定性取第一个候选值）──
        num_batches = 2
        if num_batches_input is not None and hasattr(num_batches_input, "range_values"):
            rv = num_batches_input.range_values
            if isinstance(rv, (int, float)):
                num_batches = int(rv)
            elif isinstance(rv, list) and len(rv) > 0:
                first = rv[0]
                if isinstance(first, (list, tuple)):
                    num_batches = int(first[0])
                else:
                    num_batches = int(first)
            elif isinstance(rv, str):
                num_batches = 2

        num_batches = max(1, num_batches)

        # ── 2. 解析 SeqEmb 的采样 shape ──
        in_shape = list(in_input.shape) if in_input.shape else [1024, 1024]
        jagged_size = int(in_shape[0]) if len(in_shape) > 0 else 1024
        d = int(in_shape[1]) if len(in_shape) > 1 else 1024

        # ── 3. 约束：jagged_size >= B，且 <= B * 4096（每序列 <= 4096）──
        max_total = num_batches * 4096
        jagged_size = max(1, jagged_size, num_batches)
        jagged_size = min(jagged_size, max_total)
        d = max(1, d)

        in_input.shape = [jagged_size, d]

        return case_config
