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
fa_infer_with_arbitrary_func 算子 PyTorch 接口使用样例

功能：Flash Attention 推理场景的自定义注意力掩码算子。通过函数编码张量
arbitrary_func 描述每个 Query token 允许关注的 KV 区间，并配合预先计算好的
块稀疏索引 q2k_block_sparse 跳过无关 KV 块，实现带任意区间掩码的高效注意力
前向计算。

用法:
    python3 test_torch_fa_infer_with_arbitrary_func.py
"""

import math

import torch
import torch.nn.functional as F
import ops_rec

DEVICE = "npu:0"


def create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=None):
    """根据 arbitrary_func 生成 FULL/MASK 块稀疏索引（与算子内部分类逻辑一致）。"""
    bm_B = int(func_tensor.shape[0])
    bm_H = int(func_tensor.shape[1])
    n_func = int(func_tensor.shape[2])

    if device is None:
        device = func_tensor.device

    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    mask_block_cnt = torch.zeros(bm_B, bm_H, num_q_blocks, dtype=torch.int32, device=device)
    full_block_cnt = torch.zeros(bm_B, bm_H, num_q_blocks, dtype=torch.int32, device=device)
    mask_block_idx = torch.full((bm_B, bm_H, num_q_blocks, num_kv_blocks), -1, dtype=torch.int32, device=device)
    full_block_idx = torch.full((bm_B, bm_H, num_q_blocks, num_kv_blocks), -1, dtype=torch.int32, device=device)

    kv_starts_all = torch.arange(num_kv_blocks, device=device) * KV_BLOCK_SIZE
    kv_ends_all = torch.minimum(
        kv_starts_all + KV_BLOCK_SIZE,
        torch.full((num_kv_blocks,), seqlen_k, device=device, dtype=torch.int32),
    )
    kv_block_full = (kv_ends_all - kv_starts_all) == KV_BLOCK_SIZE

    num_intervals = (n_func + 1) // 2

    for b in range(bm_B):
        for h in range(bm_H):
            for q_block in range(num_q_blocks):
                q_start = q_block * Q_BLOCK_SIZE
                q_end = min(q_start + Q_BLOCK_SIZE, seqlen_q)
                num_active = q_end - q_start
                is_q_block_full = num_active == Q_BLOCK_SIZE

                fts = func_tensor[b, h, :, q_start:q_end]

                # KV 区间范围优化：跳过整行无有效区间的 q_block
                F0 = fts[0]
                valid0 = F0 > 0
                block_min_kv = seqlen_k + 1
                block_max_kv = 0
                if valid0.any():
                    block_min_kv = 0
                    block_max_kv = int(F0[valid0].max().item())
                for ii in range(1, num_intervals):
                    starts_i = fts[2 * ii - 1]
                    ends_i = fts[2 * ii]
                    valid_i = ends_i > starts_i
                    if valid_i.any():
                        s_min = int(starts_i[valid_i].min().item())
                        e_max = int(ends_i[valid_i].max().item())
                        block_min_kv = min(block_min_kv, s_min)
                        block_max_kv = max(block_max_kv, e_max)

                if block_max_kv == 0 or block_min_kv > seqlen_k:
                    continue

                kv_block_start = block_min_kv // KV_BLOCK_SIZE
                kv_block_end = min(
                    (block_max_kv + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE,
                    num_kv_blocks,
                )
                K = kv_block_end - kv_block_start
                if K <= 0:
                    continue

                kv_starts = kv_starts_all[kv_block_start:kv_block_end].view(-1, 1)
                kv_ends = kv_ends_all[kv_block_start:kv_block_end].view(-1, 1)

                # 区间 0：[0, F0)，起始隐式为 0
                F0_col = F0.view(1, -1).expand(K, num_active)
                full_0 = F0_col >= kv_ends
                mask_0 = (F0_col > kv_starts) & (F0_col < kv_ends)
                block_full = full_0
                block_mask = mask_0

                # 剩余区间：[F_{2i-1}, F_{2i})，i = 1, ...
                for ii in range(1, num_intervals):
                    starts_col = fts[2 * ii - 1].view(1, -1).expand(K, num_active)
                    ends_col = fts[2 * ii].view(1, -1).expand(K, num_active)
                    full_i = (starts_col <= kv_starts) & (ends_col >= kv_ends)
                    mask_i = (starts_col < kv_ends) & (ends_col > kv_starts) & ~full_i
                    block_full = block_full | full_i
                    block_mask = block_mask | mask_i

                state_full = block_full.all(dim=1)
                state_any = (block_full | block_mask).any(dim=1)

                if is_q_block_full:
                    state_mask = state_any & ~state_full
                else:
                    # 部分 q_block：padding token 视为 EMPTY，块不可能为 FULL
                    state_full = torch.zeros_like(state_full)
                    state_mask = state_any

                is_full = state_full & kv_block_full[kv_block_start:kv_block_end]
                is_mask = (state_full | state_mask) & ~is_full

                full_ids = torch.nonzero(is_full, as_tuple=True)[0] + kv_block_start
                mask_ids = torch.nonzero(is_mask, as_tuple=True)[0] + kv_block_start

                full_block_cnt[b, h, q_block] = full_ids.numel()
                mask_block_cnt[b, h, q_block] = mask_ids.numel()
                if full_ids.numel() > 0:
                    full_block_idx[b, h, q_block, : full_ids.numel()] = full_ids.to(torch.int32)
                if mask_ids.numel() > 0:
                    mask_block_idx[b, h, q_block, : mask_ids.numel()] = mask_ids.to(torch.int32)

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def _build_mask(offsets_i32, total_seqlen, batch_size, ctx_lens, tgt_lens,
                target_group_size, device):
    """构建 arbitrary_func 及对应的 mask/full 块稀疏索引。"""
    af_len = total_seqlen + 256
    arbitrary_func = torch.zeros(1, 1, 5, af_len, dtype=torch.int32, device=device)
    af = arbitrary_func[0, 0, :, :total_seqlen]

    positions = torch.arange(total_seqlen, device=device, dtype=torch.int32)
    seq_id = torch.bucketize(positions, offsets_i32[1:] - 1, right=False)
    seq_start = offsets_i32[seq_id]
    seq_end = offsets_i32[seq_id + 1]
    seq_len = seq_end - seq_start

    ctx_len = ctx_lens[seq_id]
    tgt_len = tgt_lens[seq_id]
    hist_len = seq_len - ctx_len - tgt_len

    pos_in_seq = positions - seq_start
    ctx_end = ctx_len
    hist_end = ctx_len + hist_len
    base = seq_start + hist_end

    is_ctx = pos_in_seq < ctx_end
    is_hist = (pos_in_seq >= ctx_end) & (pos_in_seq < hist_end)
    is_tgt = pos_in_seq >= hist_end

    # 区间 0：[0, F0)，覆盖从序列起点到历史区间末尾
    af[1, is_ctx] = seq_start[is_ctx]
    af[2, is_ctx] = (seq_start + hist_end)[is_ctx]
    af[1, is_hist] = seq_start[is_hist]
    af[2, is_hist] = (positions + 1)[is_hist]
    tgt_idx = pos_in_seq - hist_end
    group_start = (tgt_idx // target_group_size) * target_group_size
    af[1, is_tgt] = seq_start[is_tgt]
    af[2, is_tgt] = base[is_tgt]
    af[3, is_tgt] = (base + group_start)[is_tgt]
    af[4, is_tgt] = (base + tgt_idx + 1)[is_tgt]

    mask_cnt, mask_idx, full_cnt, full_idx = create_reference_block_mask(
        arbitrary_func, total_seqlen, total_seqlen, 128, 128,
    )
    mask_offset = torch.cumsum(mask_cnt, dim=2).reshape(-1).to(mask_cnt.dtype)
    mask_offset = torch.cat([mask_offset.new_zeros(1), mask_offset])
    full_offset = torch.cumsum(full_cnt, dim=2).reshape(-1).to(full_cnt.dtype)
    full_offset = torch.cat([full_offset.new_zeros(1), full_offset])
    return arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx


def create_inputs(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    """生成 q/k/v 输入及对应的 arbitrary_func 与块稀疏索引（CPU 上完成）。"""
    total_seqlen = batch_size * (ctx_num + his_num + tgt_num)
    q_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    k_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    v_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    step = ctx_num + tgt_num + his_num
    offsets_i32 = torch.tensor([i * step for i in range(batch_size + 1)], dtype=torch.int32).to('cpu')
    ctx_lens = torch.tensor([ctx_num] * batch_size, dtype=torch.int32).to('cpu')
    tgt_lens = torch.tensor([tgt_num] * batch_size, dtype=torch.int32).to('cpu')
    target_group_size = 6
    arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx = _build_mask(
        offsets_i32, total_seqlen, batch_size, ctx_lens, tgt_lens, target_group_size, 'cpu'
    )
    mask_idx = mask_idx.flatten()
    mask_idx = mask_idx[mask_idx != -1]
    full_idx = full_idx.flatten()
    full_idx = full_idx[full_idx != -1]
    return q_b, k_b, v_b, arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx


def _apply_arbitrary_mask_cpu(qk_attn, arbitrary_func, seqlen_q, seqlen_k, device):
    """在 CPU 上应用任意区间掩码，作为参考实现。"""
    af = arbitrary_func[..., :seqlen_q]
    B, H, func_num, _ = af.shape

    j = torch.arange(seqlen_k, device=device, dtype=af.dtype).view(1, 1, 1, seqlen_k)

    base_cut = af[:, :, 0, :].unsqueeze(-1)
    base_valid = j < base_cut

    num_intervals = func_num // 2
    if num_intervals > 0:
        starts = af[:, :, 1:func_num:2, :].unsqueeze(-1)
        ends = af[:, :, 2:func_num:2, :].unsqueeze(-1)
        j_exp = j.view(1, 1, 1, 1, seqlen_k)
        in_interval = (j_exp >= starts) & (j_exp < ends)
        interval_valid = in_interval.any(dim=2)
    else:
        interval_valid = torch.zeros(B, H, seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    valid = base_valid | interval_valid
    qk_attn = torch.where(valid, qk_attn, torch.full_like(qk_attn, -float("inf")))
    return qk_attn


def compute_reference_arbitrary_cpu(q, k, v, arbitrary_func, up_cast=False):
    """CPU 参考实现：softmax 注意力（含任意区间掩码）。"""
    q = q.float() if up_cast else q
    k = k.float() if up_cast else k
    v = v.float() if up_cast else v

    total_seqlen, nheads, headdim = q.shape
    seqlen_q = total_seqlen
    seqlen_k = total_seqlen
    scale = 1.0 / math.sqrt(headdim)

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    qk_attn = torch.einsum("bnhd,bmhd->bhnm", q * scale, k)
    qk_attn = _apply_arbitrary_mask_cpu(qk_attn, arbitrary_func, seqlen_q, seqlen_k, q.device)

    softmax_attn = F.softmax(qk_attn, dim=-1)
    out = torch.einsum("bhnm,bmhd->bnhd", softmax_attn, v)
    return out.squeeze(0).contiguous()


def test_fa_infer_with_arbitrary_func():
    """验证 fa_infer_with_arbitrary_func 算子（仅前向）的推理结果。"""
    N, D = 4, 128
    ctx_num, his_num, tgt_num = 0, 1000, 3072
    batch_size = 1
    dtype = torch.bfloat16
    torch.npu.set_device(DEVICE)

    # 生成输入并构建块稀疏索引（CPU 侧完成）
    q_b, k_b, v_b, arbitrary_func, mask_cnt, mask_offset, mask_idx, \
        full_cnt, full_offset, full_idx = create_inputs(N, D, ctx_num, his_num, tgt_num, batch_size, dtype)

    q2k_block_sparse = [mask_cnt.to(DEVICE), mask_offset.to(DEVICE), mask_idx.to(DEVICE),
                        full_cnt.to(DEVICE), full_offset.to(DEVICE), full_idx.to(DEVICE)]

    # 算子调用：arbitrary_func 需裁剪为 4 个区间值并转置为 [B, H_q, s_q, 4]
    af_npu = arbitrary_func[:, :, 1:5, :].permute(0, 1, 3, 2).contiguous().to(DEVICE)
    npu_out = ops_rec.ascendc.attention.fa_infer_with_arbitrary_func(
        q_b.to(DEVICE), k_b.to(DEVICE), v_b.to(DEVICE), af_npu, q2k_block_sparse
    )

    npu_out_cpu = npu_out.to("cpu")
    cpu_out = compute_reference_arbitrary_cpu(q_b, k_b, v_b, arbitrary_func)

    atol, rtol = 5e-3, 5e-3  # bfloat16 精度阈值
    assert torch.allclose(cpu_out, npu_out_cpu, atol=atol, rtol=rtol)
    print("[torch] fa_infer_with_arbitrary_func torch example PASSED")


if __name__ == "__main__":
    test_fa_infer_with_arbitrary_func()
