#!/usr/bin/env python3
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
reverse_sequence 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_reverse_sequence.py
"""

import torch
import ops_rec


def test_reverse_sequence():
    """验证 reverse_sequence 算子的前向推理结果。"""
    bs, max_seq_len, data_dim = 4, 8, 16
    input_data = torch.randn(bs, max_seq_len, data_dim, dtype=torch.float32).npu()
    seq_lengths = torch.tensor([2, 8, 0, 5], dtype=torch.int64).npu()

    y = ops_rec.ascendc.sequence.reverse_sequence(input_data, seq_lengths)

    # 手动计算验证：每个 batch 仅前 seq_len 行逆序，其余行保持原值
    expected = input_data.clone()
    for b in range(bs):
        len_b = int(seq_lengths[b].item())
        if len_b > 0:
            expected[b, :len_b] = input_data[b, :len_b].flip(dims=[0])

    diff = (y.cpu() - expected.cpu()).abs().max().item()
    print(f"[torch] max diff vs manual: {diff:.6f}")
    assert diff < 1e-6, f"max diff {diff} exceeds 1e-6"
    print("[torch] reverse_sequence torch example PASSED")


if __name__ == "__main__":
    test_reverse_sequence()
