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
hstu_forward_v1 算子 PyTorch 接口使用样例（hstu_dense normal 格式前向）

用法:
    python3 test_torch_hstu_forward.py
"""

import torch
import torch.nn.functional as F
import ops_rec


def test_hstu_dense():
    """验证 hstu_dense 算子的前向推理结果。"""
    torch.npu.set_device(0)
    B, S, N, D = 2, 16, 4, 32
    silu_scale = 1.0 / S

    q = torch.randn(B, S, N, D, dtype=torch.float16).npu()
    k = torch.randn(B, S, N, D, dtype=torch.float16).npu()
    v = torch.randn(B, S, N, D, dtype=torch.float16).npu()

    y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, silu_scale)

    # 手动计算验证（fp32 精度）
    qk = torch.einsum("bnhd,bmhd->bhnm", q.float(), k.float())
    qk = F.silu(qk) * silu_scale
    tril = torch.tril(torch.ones(S, S, dtype=torch.float32, device="npu"))
    qk = qk * tril.unsqueeze(0).unsqueeze(0)
    expected = torch.einsum("bhnm,bmhd->bnhd", qk, v.float())

    diff = (y.float().cpu() - expected.cpu()).abs().max().item()
    print(f"[torch] max diff vs manual: {diff:.6f}")
    assert diff < 5e-2, f"max diff {diff} exceeds 5e-2"
    print("[torch] hstu_dense torch example PASSED")


if __name__ == "__main__":
    test_hstu_dense()
