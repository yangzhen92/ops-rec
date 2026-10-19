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
index_select_for_rank1_backward 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_index_select_for_rank1_backward.py
"""

import torch
import ops_rec


def test_index_select_for_rank1_backward():
    """验证 index_select_for_rank1_backward 算子的散射累加结果。"""
    x_dim0, num_idx = 16, 10
    # x 仅提供权重表 shape（grad_x 与其同形）；数值不参与计算
    x = torch.arange(x_dim0, dtype=torch.float32).npu()
    index = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=torch.int32).npu()
    grad_y = torch.arange(1, num_idx + 1, dtype=torch.float32).npu() / 4.0

    grad_x, grad_index = ops_rec.ascendc.embedding.index_select_for_rank1_backward(grad_y, x, index)

    # 手动计算验证：grad_x 清零后按 index 散射累加 grad_y
    expected = torch.zeros(x_dim0, dtype=torch.float32)
    expected.scatter_add_(0, index.cpu().to(torch.int64), grad_y.cpu())

    diff = (grad_x.cpu() - expected).abs().max().item()
    print(f"[torch] max diff vs manual: {diff:.6f}")
    assert diff < 1e-6, f"max diff {diff} exceeds 1e-6"
    # 第二个返回值为 index 梯度占位，恒为 undefined
    assert grad_index is None, "grad_index should be undefined (None)"
    print("[torch] index_select_for_rank1_backward torch example PASSED")


if __name__ == "__main__":
    test_index_select_for_rank1_backward()
