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
hstu_forward_v1 算子 PyTorch 接口使用样例

覆盖 4 种接口（参数名与 torch schema 一一对应）：
    - hstu_dense            : normal（dense）格式，q/k/v 为 4 维 [B, S, N, D]
    - hstu_jagged.equal     : jagged 格式，q 与 k/v 共用同一 seq_offset（段长一致）
    - hstu_jagged.delta     : jagged 格式，q 与 k/v 使用不同的 seq_offset / seq_offset_k（段长可不同）
    - hstu_paged            : paged 格式，kv_cache 分页存储（解码场景）

用法:
    python3 test_torch_hstu_forward.py
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


def test_hstu_dense():
    """验证 hstu_dense 算子的前向推理结果（normal 格式，内置下三角 mask）。"""
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


def test_hstu_paged():
    """paged：kv_cache 分页构造（数据构造参考 pytest test_hstu_paged_graph，已实机验证）。

    注：paged 的 numpy golden 过于复杂，此处仅验证输出 shape 与运行成功；
    数值等价性验证见 pytest（同一输入下 compiled 与 eager 输出一致）。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    page_size = 16
    batch = 2
    # 每个 batch：历史 token 数（已存入 kv_cache）与目标（新生成）token 数
    new_hist_lens = [16, 24]
    num_targets = [4, 6]

    seq_lens = [h + t for h, t in zip(new_hist_lens, num_targets)]
    total = sum(seq_lens)
    max_seq_len = max(seq_lens)

    q = torch.randn(total, N, D, dtype=torch.float16).npu()
    k = torch.randn(total, N, D, dtype=torch.float16).npu()
    v = torch.randn(total, N, D, dtype=torch.float16).npu()

    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = seq_offset.clone()
    seq_offset_t = torch.tensor([0] + [sum(num_targets[: i + 1]) for i in range(batch)], device="npu",
                                dtype=torch.int32)
    num_target = torch.tensor(num_targets, device="npu", dtype=torch.int32)

    # kv_cache 分页：每个 batch 的历史 KV 按 page_size 分页存放（末页不足时以 last_page_len 标记）
    num_pages_per_batch = [(h + page_size - 1) // page_size for h in new_hist_lens]
    page_ids = list(range(sum(num_pages_per_batch)))
    page_offsets = [0] + [sum(num_pages_per_batch[: i + 1]) for i in range(batch)]
    last_page_len = [h % page_size if h % page_size > 0 else page_size for h in new_hist_lens]

    kv_cache = torch.randn(sum(num_pages_per_batch), 2, page_size, N, D, device="npu", dtype=torch.float16)
    page_offsets = torch.tensor(page_offsets, device="npu", dtype=torch.int32)
    page_ids = torch.tensor(page_ids, device="npu", dtype=torch.int32)
    last_page_len = torch.tensor(last_page_len, device="npu", dtype=torch.int32)

    # 注意：num_target 定义时 target_group_size 必须 > 0（torch 扩展层约束）
    y = ops_rec.ascendc.attention.hstu_paged(
        q, k, v, kv_cache,
        mask=None, attn_bias=None, mask_type=0,
        max_seq_len=max_seq_len, max_seq_len_k=max_seq_len, silu_scale=1.0 / max_seq_len,
        seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
        page_offsets=page_offsets, page_ids=page_ids, last_page_len=last_page_len,
        num_target=num_target, target_group_size=1)

    assert y.shape == (total, N, D), f"unexpected output shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "output contains non-finite values"
    print(f"[torch] hstu_paged output shape: {tuple(y.shape)}")
    print("[torch] hstu_paged torch example PASSED（shape/运行成功校验，数值等价性见 pytest）")


if __name__ == "__main__":
    test_hstu_dense()
    test_hstu_jagged_equal()
    test_hstu_jagged_delta()
    test_hstu_paged()
    print("[torch] all hstu_forward_v1 torch examples PASSED")
