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
token_mixing 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_token_mixing.py
"""

import torch
import ops_rec

EPS = 1e-7


def test_token_mixing():
    """验证 token_mixing 前向结果（要求 S == H）。"""
    B, S = 2, 16
    H = S
    x = torch.randn(B, S, H, dtype=torch.float32).npu()
    gamma = torch.ones(H, dtype=torch.float32).npu()
    beta = torch.zeros(H, dtype=torch.float32).npu()

    y = ops_rec.ascendc.mixing.token_mixing(x, gamma, beta, epsilon=EPS)
    assert tuple(y.shape) == (B, H, S), f"unexpected shape {tuple(y.shape)}"

    # 手动 golden：LayerNorm(x + x^T)
    x_t = x.permute(0, 2, 1)
    add = x + x_t
    mean = add.mean(dim=-1, keepdim=True)
    var = add.var(dim=-1, keepdim=True, unbiased=False)
    expected = (add - mean) / torch.sqrt(var + EPS) * gamma + beta

    diff = (y.cpu() - expected.cpu()).abs().max().item()
    print(f"[torch] shape={tuple(y.shape)}, max diff vs manual: {diff:.6f}")
    assert diff < 1e-3, f"max diff {diff} exceeds 1e-3"
    print("[torch] token_mixing torch example PASSED")


if __name__ == "__main__":
    test_token_mixing()
