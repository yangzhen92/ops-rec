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
hstu_forward_v1 算子 PyTorch 接口使用样例 — GQA（分组查询注意力）场景

覆盖场景（参数名与 torch schema 一一对应）：
    - hstu_jagged.equal（jagged 格式）：GQA（N_q=8, N_k=2，K/V 头数小于 Q）+ qk/v dim 不等

约束：N_q % N_k == 0 且 N_k == N_v（GQA 仅 jagged 布局支持，dense 布局不支持）

用法:
    python3 test_torch_hstu_forward_gqa.py
"""

import torch
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def test_hstu_jagged_gqa():
    """运行 hstu_jagged.equal GQA 前向样例（N_q=8, N_k=2，D_qk=32, D_v=64）。

    多个 Q 头共享同一 K/V 头（每 N_q//N_k 个 Q 头共享 1 个 K/V 头），减少 KV 缓存。
    """
    torch.npu.set_device(0)
    N_q, N_k, D_qk, D_v = 8, 2, 32, 64
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    silu_scale = 1.0 / S

    q = torch.randn(total, N_q, D_qk, dtype=torch.float16).npu()
    k = torch.randn(total, N_k, D_qk, dtype=torch.float16).npu()   # N_k < N_q（GQA）
    v = torch.randn(total, N_k, D_v, dtype=torch.float16).npu()    # D_v != D_qk（dim 不等）
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.equal(q, k, v, None, None, 0, S, silu_scale, seq_offset)

    torch.npu.synchronize()
    assert y.shape == (total, N_q, D_v), f"unexpected output shape {tuple(y.shape)}"
    print(f"[torch] hstu_jagged.equal(GQA) N_q={N_q} N_k={N_k} D_qk={D_qk} D_v={D_v}, output shape: {tuple(y.shape)}")
    print("[torch] hstu_jagged GQA torch example PASSED")


if __name__ == "__main__":
    test_hstu_jagged_gqa()
    print("[torch] hstu_jagged GQA torch example done")
