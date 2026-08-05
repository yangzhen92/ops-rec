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
ln_mul 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_ln_mul.py
"""

import torch
import ops_rec


def test_ln_mul():
    """验证 ln_mul 算子的前向推理结果。"""
    A, R = 4, 8
    x = torch.randn(A, R, dtype=torch.float32).npu()
    u = torch.randn(A, R, dtype=torch.float32).npu()
    gamma = torch.ones(R, dtype=torch.float32).npu()
    beta = torch.zeros(R, dtype=torch.float32).npu()

    y = ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)

    # 手动计算验证
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)
    expected = x_norm * gamma + beta
    expected = expected * u

    diff = (y.cpu() - expected.cpu()).abs().max().item()
    print(f"[torch] max diff vs manual: {diff:.6f}")
    assert diff < 1e-3, f"max diff {diff} exceeds 1e-3"
    print("[torch] ln_mul torch example PASSED")


if __name__ == "__main__":
    test_ln_mul()
