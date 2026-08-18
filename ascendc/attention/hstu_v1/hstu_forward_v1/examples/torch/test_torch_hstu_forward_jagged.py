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
hstu_forward_v1 算子 PyTorch 接口使用样例 — jagged 格式

覆盖接口（参数名与 torch schema 一一对应）：
    - hstu_jagged.equal     : jagged 格式，q 与 k/v 共用同一 seq_offset（段长一致）
    - hstu_jagged.delta     : jagged 格式，q 与 k/v 使用不同的 seq_offset / seq_offset_k（段长可不同）

用法:
    python3 test_torch_hstu_forward_jagged.py
"""

import torch
import torch.nn.functional as F
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def _hstu_jagged_golden(q, k, v, seq_offset, seq_offset_k=None, silu_scale=1.0, mask_type=0):
    """CPU 侧 jagged HSTU golden（fp32 精度），与 pytest 的 get_golden_jagged 逻辑一致。

    delta 语义（seq_offset_k 与 seq_offset 不同）下 q 段长 S_q 与 k 段长 S_k 可以不同，
    qk 形状为 (S_q, S_k)；mask_type=0 时按下三角（对角偏移 S_k - S_q）裁剪。
    """
    if seq_offset_k is None:
        seq_offset_k = seq_offset
    outs = []
    for b in range(seq_offset.numel() - 1):
        q0, q1 = int(seq_offset[b]), int(seq_offset[b + 1])
        k0, k1 = int(seq_offset_k[b]), int(seq_offset_k[b + 1])
        qb, kb, vb = q[q0:q1], k[k0:k1], v[k0:k1]
        s_q, s_k = qb.shape[0], kb.shape[0]
        qk = torch.einsum("nhd,mhd->hnm", qb, kb)
        qk = F.silu(qk) * silu_scale
        if mask_type == 0:
            tril = torch.tril(torch.ones(s_q, s_k, dtype=torch.float32), diagonal=s_k - s_q)
            qk = qk * tril.unsqueeze(0)
        outs.append(torch.einsum("hnm,mhd->nhd", qk, vb))
    return torch.cat(outs, dim=0)


def test_hstu_jagged_equal():
    """jagged equal：q 与 k/v 共用同一 seq_offset，段长一致。

    数据构造与 golden 验证参考 pytest test_hstu_jagged_graph。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    silu_scale = 1.0 / S

    q = torch.randn(total, N, D, dtype=torch.float16).npu()
    k = torch.randn(total, N, D, dtype=torch.float16).npu()
    v = torch.randn(total, N, D, dtype=torch.float16).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.equal(q, k, v, None, None, 0, S, silu_scale, seq_offset)

    # 手动计算验证：按 seq_offset 拆段，每段做 silu(qk^T)*scale @ v（fp32 精度）
    expected = _hstu_jagged_golden(q.float().cpu(), k.float().cpu(), v.float().cpu(),
                                   seq_offset.cpu(), silu_scale=silu_scale)
    diff = (y.float().cpu() - expected).abs().max().item()
    print(f"[torch] jagged equal max diff vs manual: {diff:.6f}")
    assert diff < 5e-2, f"max diff {diff} exceeds 5e-2"
    print("[torch] hstu_jagged.equal torch example PASSED")


def test_hstu_jagged_delta():
    """jagged delta：q 与 k/v 使用不同的 seq_offset / seq_offset_k，段长可不同。

    数据构造与 golden 验证参考 pytest test_hstu_jagged_delta_graph。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens_q = [4, 6]
    seq_lens_k = [6, 8]
    batch = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    s_q = max(seq_lens_q)
    s_k = max(seq_lens_k)
    silu_scale = 1.0 / s_q

    q = torch.randn(total_q, N, D, dtype=torch.float16).npu()
    k = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    v = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens_q[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = torch.tensor([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch)], device="npu",
                                dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.delta(q, k, v, None, None, 0, s_q, s_k, silu_scale,
                                                    seq_offset, seq_offset_k)

    # 手动计算验证：q 段长与 k 段长不同，qk 形状为 (S_q, S_k)（fp32 精度）
    expected = _hstu_jagged_golden(q.float().cpu(), k.float().cpu(), v.float().cpu(),
                                   seq_offset.cpu(), seq_offset_k.cpu(), silu_scale)
    diff = (y.float().cpu() - expected).abs().max().item()
    print(f"[torch] jagged delta max diff vs manual: {diff:.6f}")
    assert diff < 5e-2, f"max diff {diff} exceeds 5e-2"
    print("[torch] hstu_jagged.delta torch example PASSED")


if __name__ == "__main__":
    test_hstu_jagged_equal()
    test_hstu_jagged_delta()
    print("[torch] hstu_jagged torch examples done")
