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

import sys
import pytest
import torch
import numpy as np

import ops_rec

torch.npu.config.allow_internal_format = False


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def get_golden(q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=None, silu_scale=None):
    """numpy 实现 HSTU 前向：out = (silu(qk^T + bias) * silu_scale * mask) @ v（fp32 精度）

    参数顺序与算子接口 hstu_dense(q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale) 对齐。
    """
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    B, S, N_q, D = q.shape
    scale = silu_scale if silu_scale not in (None, 0.0) else 1.0 / max_seq_len

    # GQA：N_q != N_k 时，k/v 的 head 维按 N_q // N_k 扩展（repeat_interleave 语义，
    # q head i 共享 k/v head i // ratio），与 RecSDK golden（test_hstu_jagged_forward.py
    # golden_op_exec）一致；dim 维不扩展。N_q == N_k 时行为不变。
    N_k = k.shape[2]
    if N_k != N_q:
        assert N_q % N_k == 0, f"head_nums_q ({N_q}) must be divisible by head_nums_k ({N_k})"
        h_qk_ratio = N_q // N_k
        k_f = np.repeat(k_f, h_qk_ratio, axis=2)
        v_f = np.repeat(v_f, h_qk_ratio, axis=2)

    qk = np.einsum("bnhd,bmhd->bhnm", q_f, k_f)
    if attn_bias is not None:
        qk = qk + attn_bias.astype(np.float32)
    qk = qk * (1.0 / (1.0 + np.exp(-qk))) * scale  # silu
    if mask is not None:
        qk = qk * mask.astype(np.float32)
    elif mask_type == 0:
        qk = qk * np.tril(np.ones((S, S), dtype=np.float32))[None, None, :, :]
    out = np.einsum("bhnm,bmhd->bnhd", qk, v_f)
    return out.astype(q.dtype)


def get_golden_jagged(q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=None,
                      silu_scale=None, seq_offset=None, seq_offset_k=None):
    """numpy 实现 HSTU jagged 前向：按 batch 拆分 q 段（seq_offset）与 k 段（seq_offset_k）后每段做 silu(qk^T)*scale @ v

    参数顺序与算子接口 hstu_jagged.equal/delta 的公共部分 (q, k, v, mask, attn_bias, mask_type,
    max_seq_len, silu_scale, seq_offset, seq_offset_k) 对齐；mask/attn_bias/max_seq_len 仅为
    接口对齐保留（本函数内部逻辑不变，mask 由 mask_type 控制）。

    delta 语义（seq_offset_k 与 seq_offset 不同）下 q 段长 S_q 与 k 段长 S_k 可以不同，
    qk 形状为 (S_q, S_k)；mask_type=0 时按内核语义使用下三角（对角偏移 S_k - S_q）裁剪。
    """
    if seq_offset is None:
        raise ValueError("get_golden_jagged: seq_offset is required")
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    if seq_offset_k is None:
        seq_offset_k = seq_offset
    outs = []
    for b in range(len(seq_offset) - 1):
        q0, q1 = seq_offset[b], seq_offset[b + 1]
        k0, k1 = seq_offset_k[b], seq_offset_k[b + 1]
        qb, kb, vb = q_f[q0:q1], k_f[k0:k1], v_f[k0:k1]
        S_q, S_k = qb.shape[0], kb.shape[0]
        qk = np.einsum("nhd,mhd->hnm", qb, kb)
        qk = qk * (1.0 / (1.0 + np.exp(-qk))) * silu_scale  # silu
        if mask_type == 0:
            qk = qk * np.tril(np.ones((S_q, S_k), dtype=np.float32), k=S_k - S_q)[None, :, :]
        outs.append(np.einsum("hnm,mhd->nhd", qk, vb))
    return np.concatenate(outs, axis=0).astype(q.dtype)


def get_golden_paged(q, k, v, kv_cache, seq_offset, page_offsets, page_ids, last_page_len,
                     num_target, silu_scale):
    """numpy 实现 HSTU paged 前向 golden（fp32），针对 MASK_TRIL（mask_type=0）。

    语义依据 op_kernel/atlas950/hstu_paged_forward_kernel.h（FetchKvMayFromCache 的
    cache/input 拼接）+ hstu_dense_causal_mask.h（GenCausalMask/GenTargetMask），并与
    RecSDK cust_op/test/hstu_dense/torch/test_hstu_paged_forward.py 参考 golden 交叉确认：
      - q 行 = 每个 batch 的全部 token（历史 + target，由 seq_offset 分段），输出 shape (total, N, D)
      - K/V 列 = kv_cache 解页后的历史 KV（H 行，由 page_offsets/page_ids/last_page_len 索引）
        + 输入 k/v 的 target 段（T 行），共 H+T 行；输入 k/v 中的历史段不作为 KV 使用
      - mask：历史行 i 仅见历史列 [0, i]（下三角）；target 行 H+j 见全部历史列 [0, H)
        + 自身列 H+j（eye，target 之间互不 attend；即 GenCausalMask 下三角再被
        GenTargetMask 挖空 [H, H+j)，与 RecSDK golden 的 tril+eye 一致）
      - 输出 = silu(qk) * silu_scale * mask @ v_con，按 q 的 token 顺序拼接
    """
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    outs = []
    for b in range(len(seq_offset) - 1):
        q0, q1 = seq_offset[b], seq_offset[b + 1]
        S = int(q1 - q0)                       # 该 batch 全部 q 行（历史 + target）
        T = int(num_target[b])
        H = S - T                              # 历史行/列数 == kv_cache 覆盖长度
        # 1) 解页 kv_cache：每 batch 的 page 依序拼接，末页只取 last_page_len 行
        cache_k, cache_v = [], []
        for p in range(int(page_offsets[b]), int(page_offsets[b + 1])):
            pid = int(page_ids[p])
            is_last = (p == int(page_offsets[b + 1]) - 1)
            n = int(last_page_len[b]) if is_last else kv_cache.shape[2]
            cache_k.append(kv_cache[pid, 0, :n])
            cache_v.append(kv_cache[pid, 1, :n])
        cache_k = np.concatenate(cache_k, axis=0)   # (H, N, D)
        cache_v = np.concatenate(cache_v, axis=0)
        assert len(cache_k) == H, \
            f"batch {b}: cache len {len(cache_k)} != H={H} (seq_offset 与 page 元数据不一致)"
        # 2) K/V 全序列 = cache 历史 + 输入 k/v 的 target 段（第 H..S 行）
        k_con = np.concatenate([cache_k, k_f[q0 + H:q1]], axis=0)   # (H+T, N, D)
        v_con = np.concatenate([cache_v, v_f[q0 + H:q1]], axis=0)
        # 3) 注意力 + 掩码
        qb = q_f[q0:q1]
        qk = np.einsum("nhd,mhd->hnm", qb, k_con)
        qk = qk * (1.0 / (1.0 + np.exp(-qk))) * silu_scale
        mask = np.zeros((S, S), dtype=np.float32)
        mask[:, :H] = np.tril(np.ones((S, H), dtype=np.float32), k=H + T - S)  # 历史列：下三角
        mask[H:, H:] = np.eye(T, dtype=np.float32)                            # target 列：自身
        qk = qk * mask[None, :, :]
        outs.append(np.einsum("hnm,mhd->nhd", qk, v_con))
    return np.concatenate(outs, axis=0).astype(q.dtype)


def get_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 2e-2, 2e-2  # fp16/bf16 累加误差放宽


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32), (2, 32, 4, 32)])
def test_hstu_dense_eager(dtype, shape):
    """eager 模式：MHA，mask_type=0 内置下三角"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None,
                                                mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_gqa_dimmismatch(dtype, shape):
    """GQA（N_k < N_q）且 qk/v dim 不等"""
    torch.npu.set_device(0)
    B, S, N_q, D_qk = shape
    N_k, D_v = 2, 64
    q = torch.randn(B, S, N_q, D_qk, device="npu", dtype=dtype)
    k = torch.randn(B, S, N_k, D_qk, device="npu", dtype=dtype)
    v = torch.randn(B, S, N_k, D_v, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None,
                                                mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    assert y_op.shape == (B, S, N_q, D_v)
    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_custom_mask(dtype, shape):
    """mask_type=3：自定义 mask（下三角 0/1）"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)
    tril = np.tril(np.ones((S, S), dtype=np.float32))
    mask = torch.from_numpy(np.broadcast_to(tril, (B, N, S, S))).to(device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=mask.cpu().numpy(), mask_type=3,
                          max_seq_len=S, silu_scale=1.0 / S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask, None, 3,
                                                max_seq_len=S, silu_scale=1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_hstu_jagged_equal_eager(dtype):
    """eager 模式：jagged（equal）独立精度验证（段 [4,6]，数据构造同 test_hstu_jagged_graph）"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    q = torch.randn(total, N, D, device="npu", dtype=dtype)
    k = torch.randn(total, N, D, device="npu", dtype=dtype)
    v = torch.randn(total, N, D, device="npu", dtype=dtype)
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)],
                              device="npu", dtype=torch.int32)

    y_golden = get_golden_jagged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                 mask=None, attn_bias=None, mask_type=0,
                                 max_seq_len=S, silu_scale=1.0 / S,
                                 seq_offset=seq_offset.cpu().numpy())
    y_op = ops_rec.ascendc.attention.hstu_jagged.equal(
        q, k, v, mask=None, attn_bias=None, mask_type=0,
        max_seq_len=S, silu_scale=1.0 / S, seq_offset=seq_offset)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_hstu_jagged_delta_eager(dtype):
    """eager 模式：jagged（delta）独立精度验证（q 段 [4,6] / k 段 [6,8]，数据构造同 delta_graph）"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens_q = [4, 6]
    seq_lens_k = [6, 8]
    batch = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    S_q = max(seq_lens_q)
    S_k = max(seq_lens_k)
    q = torch.randn(total_q, N, D, device="npu", dtype=dtype)
    k = torch.randn(total_k, N, D, device="npu", dtype=dtype)
    v = torch.randn(total_k, N, D, device="npu", dtype=dtype)
    seq_offset = torch.tensor([0] + [sum(seq_lens_q[: i + 1]) for i in range(batch)],
                              device="npu", dtype=torch.int32)
    seq_offset_k = torch.tensor([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch)],
                                device="npu", dtype=torch.int32)

    y_golden = get_golden_jagged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                 mask=None, attn_bias=None, mask_type=0,
                                 max_seq_len=S_q, silu_scale=1.0 / S_q,
                                 seq_offset=seq_offset.cpu().numpy(),
                                 seq_offset_k=seq_offset_k.cpu().numpy())
    y_op = ops_rec.ascendc.attention.hstu_jagged.delta(
        q, k, v, mask=None, attn_bias=None, mask_type=0,
        max_seq_len=S_q, max_seq_len_k=S_k, silu_scale=1.0 / S_q,
        seq_offset=seq_offset, seq_offset_k=seq_offset_k)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_graph(dtype, shape):
    """inductor 图模式（Meta 注册走通构图）"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_dense, dynamic=True, fullgraph=True,
                                 backend="inductor")
    y_op = compiled_fn(q=q, k=k, v=v, mask=None, attn_bias=None,
                       mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
def test_hstu_jagged_graph(dtype):
    """inductor 图模式：jagged（equal）前向入图"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    q = torch.randn(total, N, D, device="npu", dtype=dtype)
    k = torch.randn(total, N, D, device="npu", dtype=dtype)
    v = torch.randn(total, N, D, device="npu", dtype=dtype)
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y_golden = get_golden_jagged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                 mask=None, attn_bias=None, mask_type=0,
                                 max_seq_len=S, silu_scale=1.0 / S,
                                 seq_offset=seq_offset.cpu().numpy())
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_jagged.equal, dynamic=True, fullgraph=True,
                                 backend="inductor")
    y_op = compiled_fn(q, k, v, mask=None, attn_bias=None, mask_type=0,
                       max_seq_len=S, silu_scale=1.0 / S, seq_offset=seq_offset)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
def test_hstu_jagged_delta_graph(dtype):
    """inductor 图模式：jagged（delta，q 与 k/v 使用不同的 seq_offset，段长可不同）前向入图"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens_q = [4, 6]
    seq_lens_k = [6, 8]
    batch = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    S_q = max(seq_lens_q)
    S_k = max(seq_lens_k)
    q = torch.randn(total_q, N, D, device="npu", dtype=dtype)
    k = torch.randn(total_k, N, D, device="npu", dtype=dtype)
    v = torch.randn(total_k, N, D, device="npu", dtype=dtype)
    seq_offset = torch.tensor([0] + [sum(seq_lens_q[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = torch.tensor([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y_golden = get_golden_jagged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                 mask=None, attn_bias=None, mask_type=0,
                                 max_seq_len=S_q, silu_scale=1.0 / S_q,
                                 seq_offset=seq_offset.cpu().numpy(),
                                 seq_offset_k=seq_offset_k.cpu().numpy())
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_jagged.delta, dynamic=True, fullgraph=True,
                                 backend="inductor")
    y_op = compiled_fn(q, k, v, mask=None, attn_bias=None, mask_type=0,
                       max_seq_len=S_q, max_seq_len_k=S_k, silu_scale=1.0 / S_q,
                       seq_offset=seq_offset, seq_offset_k=seq_offset_k)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
def test_hstu_paged_graph(dtype):
    """inductor 图模式：paged（kv_cache 分页解码）前向，compiled 与 eager 结果一致

    paged 的 numpy golden 过于复杂，此处验证重点为：inductor 构图（Meta 走通）
    + 实机执行与 eager 等价（同一输入分别跑 torch.compile 版与 eager 版）。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    page_size = 32  # tiling 约束：page_size 必须 ∈ {32, 64, 128, 256}
    batch = 2
    # 每个 batch：历史 token 数（已存入 kv_cache）与目标（新生成）token 数
    new_hist_lens = [32, 48]  # 覆盖恰好满页(32)与末页不足(48 % 32 = 16)两种情形
    num_targets = [4, 6]

    seq_lens = [h + t for h, t in zip(new_hist_lens, num_targets)]
    total = sum(seq_lens)
    max_seq_len_q = max(seq_lens)
    max_seq_len_k = max(seq_lens)

    q = torch.randn(total, N, D, device="npu", dtype=dtype)
    k = torch.randn(total, N, D, device="npu", dtype=dtype)
    v = torch.randn(total, N, D, device="npu", dtype=dtype)

    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = seq_offset.clone()
    seq_offset_t = torch.tensor([0] + [sum(num_targets[: i + 1]) for i in range(batch)], device="npu",
                                dtype=torch.int32)
    num_target = torch.tensor(num_targets, device="npu", dtype=torch.int32)

    # kv_cache 分页：每个 batch 的历史 KV 按 page_size 分页存放（末页不足时以 last_page_len 标记）
    # 注意：page_size 必须 ∈ {32, 64, 128, 256}（tiling 层 BLOCK_HEIGHT=256 / MIN_PAGE_SIZE=32 约束）
    num_pages_per_batch = [(h + page_size - 1) // page_size for h in new_hist_lens]
    page_ids = list(range(sum(num_pages_per_batch)))
    page_offsets = [0] + [sum(num_pages_per_batch[: i + 1]) for i in range(batch)]
    last_page_len = [h % page_size if h % page_size > 0 else page_size for h in new_hist_lens]

    kv_cache = torch.randn(sum(num_pages_per_batch), 2, page_size, N, D, device="npu", dtype=dtype)
    page_offsets_t = torch.tensor(page_offsets, device="npu", dtype=torch.int32)
    page_ids_t = torch.tensor(page_ids, device="npu", dtype=torch.int32)
    last_page_len_t = torch.tensor(last_page_len, device="npu", dtype=torch.int32)

    # 注意：num_target 定义时 target_group_size 必须 > 0（TORCH_CHECK 约束）；
    # kv_cache 是 schema 第 4 位置参数，其余可选参数用显式 kwargs（与 schema 顺序一致）
    y_eager = ops_rec.ascendc.attention.hstu_paged(
        q, k, v, kv_cache,
        mask=None, attn_bias=None, mask_type=0,
        max_seq_len=max_seq_len_q, max_seq_len_k=max_seq_len_k,
        silu_scale=1.0 / max_seq_len_q,
        seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
        page_offsets=page_offsets_t, page_ids=page_ids_t, last_page_len=last_page_len_t,
        num_target=num_target, target_group_size=1)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_paged, dynamic=True, fullgraph=True,
                                 backend="inductor")
    y_op = compiled_fn(
        q, k, v, kv_cache,
        mask=None, attn_bias=None, mask_type=0,
        max_seq_len=max_seq_len_q, max_seq_len_k=max_seq_len_k,
        silu_scale=1.0 / max_seq_len_q,
        seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
        page_offsets=page_offsets_t, page_ids=page_ids_t, last_page_len=last_page_len_t,
        num_target=num_target, target_group_size=1)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_eager.cpu().numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_hstu_paged_eager(dtype):
    """eager 模式：paged 独立精度验证（真实 numpy golden，非 graph-vs-eager 对比）

    paged 语义（读 hstu_paged_forward_kernel.h / hstu_dense_causal_mask.h，并与 RecSDK
    cust_op/test/hstu_dense/torch/test_hstu_paged_forward.py 参考 golden 交叉确认）：
      - q 行 = 每个 batch 全部 token（历史 + target，seq_offset 段），输出 shape (total, N, D)
      - K/V 列 = kv_cache 解页历史 KV（H 行）+ 输入 k/v 的 target 段（T 行），共 H+T 行
      - MASK_TRIL（num_context=0, target_group_size=1）：
          hist 行 i 仅见历史列 [0, i]；target 行 H+j 见全部历史列 [0, H) + 自身列 H+j（eye）
      验证逻辑见 get_golden_paged。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    page_size = 32  # tiling 约束：page_size 必须 ∈ {32, 64, 128, 256}
    batch = 2
    new_hist_lens = [32, 48]  # 历史 token 数（已存入 kv_cache）：恰好满页(32) 与末页不足(48 % 32 = 16)
    num_targets = [4, 6]

    seq_lens = [h + t for h, t in zip(new_hist_lens, num_targets)]
    total = sum(seq_lens)
    max_seq_len_q = max(seq_lens)
    max_seq_len_k = max(seq_lens)

    q = torch.randn(total, N, D, device="npu", dtype=dtype)
    k = torch.randn(total, N, D, device="npu", dtype=dtype)
    v = torch.randn(total, N, D, device="npu", dtype=dtype)

    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)],
                              device="npu", dtype=torch.int32)
    seq_offset_k = seq_offset.clone()
    seq_offset_t = torch.tensor([0] + [sum(num_targets[: i + 1]) for i in range(batch)],
                                device="npu", dtype=torch.int32)
    num_target = torch.tensor(num_targets, device="npu", dtype=torch.int32)

    # kv_cache 分页：每个 batch 的历史 KV 按 page_size 分页存放（末页不足时以 last_page_len 标记）
    num_pages_per_batch = [(h + page_size - 1) // page_size for h in new_hist_lens]
    page_ids = list(range(sum(num_pages_per_batch)))
    page_offsets = [0] + [sum(num_pages_per_batch[: i + 1]) for i in range(batch)]
    last_page_len = [h % page_size if h % page_size > 0 else page_size for h in new_hist_lens]

    kv_cache = torch.randn(sum(num_pages_per_batch), 2, page_size, N, D, device="npu", dtype=dtype)
    page_offsets_t = torch.tensor(page_offsets, device="npu", dtype=torch.int32)
    page_ids_t = torch.tensor(page_ids, device="npu", dtype=torch.int32)
    last_page_len_t = torch.tensor(last_page_len, device="npu", dtype=torch.int32)

    y_golden = get_golden_paged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                kv_cache.cpu().numpy(),
                                seq_offset=seq_offset.cpu().numpy(),
                                page_offsets=page_offsets_t.cpu().numpy(),
                                page_ids=page_ids_t.cpu().numpy(),
                                last_page_len=last_page_len_t.cpu().numpy(),
                                num_target=num_target.cpu().numpy(),
                                silu_scale=1.0 / max_seq_len_q)
    # kv_cache 是 schema 第 4 位置参数，其余可选参数用显式 kwargs（与 schema 顺序一致）
    y_op = ops_rec.ascendc.attention.hstu_paged(
        q, k, v, kv_cache,
        mask=None, attn_bias=None, mask_type=0,
        max_seq_len=max_seq_len_q, max_seq_len_k=max_seq_len_k,
        silu_scale=1.0 / max_seq_len_q,
        seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
        page_offsets=page_offsets_t, page_ids=page_ids_t, last_page_len=last_page_len_t,
        num_target=num_target, target_group_size=1)

    assert y_op.shape == (total, N, D)
    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.ci
def test_hstu_dense_ci():
    """CI 快速冒烟用例：固定 shape + float16，仅验证正确性"""
    torch.npu.set_device(0)
    B, S, N, D = 1, 16, 4, 32
    q = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None,
                                                mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
