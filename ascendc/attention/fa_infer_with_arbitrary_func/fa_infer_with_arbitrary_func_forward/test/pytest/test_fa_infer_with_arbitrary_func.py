import sys
import math

import torch
import pytest
import torch.nn.functional as F
import ops_rec
from einops import repeat

DEVICE = "npu:0"
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def _build_mask_python(
    offsets_i32: torch.Tensor,
    total_seqlen: int,
    batch_size: int,
    ctx_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
    target_group_size: int,
    device: torch.device,
):
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

    return arbitrary_func


def create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=None):
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

    # Per-kv-block metadata (invariant across (b, h, q_block)).
    kv_starts_all = torch.arange(num_kv_blocks, device=device) * KV_BLOCK_SIZE  # [K]
    kv_ends_all = torch.minimum(
        kv_starts_all + KV_BLOCK_SIZE,
        torch.full((num_kv_blocks,), seqlen_k, device=device, dtype=torch.int32),
    )  # [K]
    # kv_block_full[k]: is the k-th block a full-sized one (i.e. not the
    # partial trailing block when seqlen_k % KV_BLOCK_SIZE != 0)?
    kv_block_full = (kv_ends_all - kv_starts_all) == KV_BLOCK_SIZE  # [K]

    num_intervals = (n_func + 1) // 2  # n_func is odd (1, 3, 5, ...)

    for b in range(bm_B):
        for h in range(bm_H):
            for q_block in range(num_q_blocks):
                q_start = q_block * Q_BLOCK_SIZE
                q_end = min(q_start + Q_BLOCK_SIZE, seqlen_q)
                num_active = q_end - q_start
                is_q_block_full = num_active == Q_BLOCK_SIZE

                # Slice func values for the active q tokens in this block.
                fts = func_tensor[b, h, :, q_start:q_end]  # [n_func, num_active]

                # --- KV range optimisation (mirrors the kernel's
                # reduce_kv_block_range).  Skip the row entirely when no q
                # token has any valid kv interval.
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
                    continue  # All empty for this (b, h, q_block)

                kv_block_start = block_min_kv // KV_BLOCK_SIZE
                kv_block_end = min(
                    (block_max_kv + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE,
                    num_kv_blocks,
                )
                K = kv_block_end - kv_block_start
                if K <= 0:
                    continue

                # --- Vectorised per-kv-block classification
                # Shape convention: [K, num_active] — one row per kv_block,
                # one column per active q_token.  Broadcasting fts (squeezed
                # to [1, num_active]) against kv_starts_all/ends_all sliced
                # to [K, 1] gives us all (kv_block, q) pairs at once.  This
                # mirrors the kernel's get_token_kv_block_state but
                # vectorises away the per-kv_block Python loop.
                kv_starts = kv_starts_all[kv_block_start:kv_block_end].view(-1, 1)  # [K, 1]
                kv_ends = kv_ends_all[kv_block_start:kv_block_end].view(-1, 1)  # [K, 1]

                # Interval 0: [0, F0) — start is implicitly 0.
                F0_col = F0.view(1, -1).expand(K, num_active)  # [K, num_active]
                full_0 = F0_col >= kv_ends  # [K, num_active]
                mask_0 = (F0_col > kv_starts) & (F0_col < kv_ends)
                block_full = full_0
                block_mask = mask_0

                # Remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, ...
                for ii in range(1, num_intervals):
                    starts_col = fts[2 * ii - 1].view(1, -1).expand(K, num_active)
                    ends_col = fts[2 * ii].view(1, -1).expand(K, num_active)
                    full_i = (starts_col <= kv_starts) & (ends_col >= kv_ends)
                    mask_i = (starts_col < kv_ends) & (ends_col > kv_starts) & ~full_i
                    block_full = block_full | full_i
                    block_mask = block_mask | mask_i

                # --- Reduce across q tokens
                # The kernel treats padding tokens (tid >= num_active) as
                # STATE_EMPTY when check_q_boundary=True.  We never materialise
                # those tokens — instead we encode the same effect at the
                # block level below.
                state_full = block_full.all(dim=1)  # [K]
                state_any = (block_full | block_mask).any(dim=1)  # [K]

                if is_q_block_full:
                    # Full q_block: MASK iff not all FULL and some valid.
                    state_mask = state_any & ~state_full  # [K]
                else:
                    # Partial q_block: padding tokens are EMPTY, so the block
                    # can't be FULL and is MASK whenever any active token sees
                    # it as valid (FULL|EMPTY -> MASK, EMPTY|EMPTY -> EMPTY).
                    state_full = torch.zeros_like(state_full)
                    state_mask = state_any

                # is_full[k] = state_full[k] AND kv_block_full[k]
                # is_mask[k] = (state_full[k] OR state_mask[k]) AND NOT is_full[k]
                is_full = state_full & kv_block_full[kv_block_start:kv_block_end]
                is_mask = (state_full | state_mask) & ~is_full

                # --- Write outputs (ascending block ids)
                full_ids = torch.nonzero(is_full, as_tuple=True)[0] + kv_block_start
                mask_ids = torch.nonzero(is_mask, as_tuple=True)[0] + kv_block_start

                full_block_cnt[b, h, q_block] = full_ids.numel()
                mask_block_cnt[b, h, q_block] = mask_ids.numel()
                if full_ids.numel() > 0:
                    full_block_idx[b, h, q_block, : full_ids.numel()] = full_ids.to(torch.int32)
                if mask_ids.numel() > 0:
                    mask_block_idx[b, h, q_block, : mask_ids.numel()] = mask_ids.to(torch.int32)

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def _build_mask(
    offsets_i32: torch.Tensor,
    total_seqlen: int,
    batch_size: int,
    ctx_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
    target_group_size: int,
    device: torch.device,
    normal=True,
):
    arbitrary_func = _build_mask_python(
        offsets_i32, total_seqlen, batch_size, ctx_lens, tgt_lens, target_group_size, device
    )

    mask_cnt, mask_idx, full_cnt, full_idx = create_reference_block_mask(
        arbitrary_func,
        total_seqlen,
        total_seqlen,
        128,
        128,
    )
    mask_offset = torch.cumsum(mask_cnt, dim=2).reshape(-1).to(mask_cnt.dtype)
    mask_offset = torch.cat([mask_offset.new_zeros(1), mask_offset])
    full_offset = torch.cumsum(full_cnt, dim=2).reshape(-1).to(full_cnt.dtype)
    full_offset = torch.cat([full_offset.new_zeros(1), full_offset])
    return arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx


def create_inputs(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    total_seqlen = batch_size * (ctx_num + his_num + tgt_num)
    q_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    k_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    v_b = torch.rand(total_seqlen, N, D, dtype=dtype, device='cpu').uniform_(-1, 1)
    step = ctx_num + tgt_num + his_num
    offsets_i32 = torch.tensor([i * step for i in range(batch_size + 1)], dtype=torch.int32).to('cpu')
    ctx_lens = torch.tensor([ctx_num] * batch_size, dtype=torch.int32).to('cpu')
    tgt_lens = torch.tensor([tgt_num] * batch_size, dtype=torch.int32).to('cpu')
    target_group_size = 6  # 默认为6，可以调整
    arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx = _build_mask(
        offsets_i32, total_seqlen, batch_size, ctx_lens, tgt_lens, target_group_size, 'cpu'
    )
    mask_idx = mask_idx.flatten()
    mask_idx = mask_idx[mask_idx != -1]
    full_idx = full_idx.flatten()
    full_idx = full_idx[full_idx != -1]
    return q_b, k_b, v_b, arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx


def create_init_inputs(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    """初始化测试输入，并构建搬移到 NPU 上的 q2k_block_sparse 稀疏索引。"""
    q_b, k_b, v_b, arbitrary_func, mask_cnt, mask_offset, mask_idx, full_cnt, full_offset, full_idx = create_inputs(
        N, D, ctx_num, his_num, tgt_num, batch_size, dtype
    )
    q2k_block_sparse = [
        mask_cnt.to(DEVICE),
        mask_offset.to(DEVICE),
        mask_idx.to(DEVICE),
        full_cnt.to(DEVICE),
        full_offset.to(DEVICE),
        full_idx.to(DEVICE),
    ]
    atol = 1e-4
    rtol = 1e-4
    if dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    elif dtype == torch.bfloat16:
        atol = 5e-3
        rtol = 5e-3
    return q_b, k_b, v_b, arbitrary_func, q2k_block_sparse, atol, rtol


def fa_infer_with_arbitrary_func_fused(query, key, value, arbitrary_func, q2k_block_sparse):
    # [batch, seqlen, headSize, headDim] -> [-1, headSize, headDim]
    torch.npu.set_device(DEVICE)
    arbitrary_func = arbitrary_func[:, :, 1:5, :].permute(0, 1, 3, 2).contiguous()
    out = ops_rec.ascendc.attention.fa_infer_with_arbitrary_func(
        query.to(DEVICE), key.to(DEVICE), value.to(DEVICE), arbitrary_func.to(DEVICE), q2k_block_sparse
    )
    return out


def _apply_arbitrary_mask_cpu(qk_attn, arbitrary_func, seqlen_q, seqlen_k, device):
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
    q = q.float() if up_cast else q
    k = k.float() if up_cast else k
    v = v.float() if up_cast else v

    total_seqlen, nheads, headdim = q.shape
    seqlen_q = total_seqlen
    seqlen_k = total_seqlen
    nheads_kv = k.shape[1]
    scale = 1.0 / math.sqrt(headdim)

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    if nheads_kv != nheads:
        k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)

    qk_attn = torch.einsum("bnhd,bmhd->bhnm", q * scale, k)
    qk_attn = _apply_arbitrary_mask_cpu(qk_attn, arbitrary_func, seqlen_q, seqlen_k, q.device)

    softmax_attn = F.softmax(qk_attn, dim=-1)
    out = torch.einsum("bhnm,bmhd->bnhd", softmax_attn, v)
    out = out.squeeze(0).contiguous()
    return out


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("ctx_num", [0, 128])
@pytest.mark.parametrize("his_num", [500, 1000])
@pytest.mark.parametrize("tgt_num", [1024, 3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_fa_infer_with_arbitrary_func_normal(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    q_b, k_b, v_b, arbitrary_func, q2k_block_sparse, atol, rtol = create_init_inputs(
        N, D, ctx_num, his_num, tgt_num, batch_size, dtype
    )
    npu_out = fa_infer_with_arbitrary_func_fused(q_b, k_b, v_b, arbitrary_func, q2k_block_sparse)
    npu_out_cpu = npu_out.to("cpu")
    # float16 CPU执行耗时较长, 转fp32后计算再验证
    cpu_out = compute_reference_arbitrary_cpu(q_b, k_b, v_b, arbitrary_func, dtype == torch.float16)
    assert torch.allclose(cpu_out.to(dtype), npu_out_cpu, atol=atol, rtol=rtol)


@pytest.mark.ci
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("ctx_num", [0])
@pytest.mark.parametrize("his_num", [1000])
@pytest.mark.parametrize("tgt_num", [3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1])
def test_fa_infer_with_arbitrary_func_ci(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    q_b, k_b, v_b, arbitrary_func, q2k_block_sparse, atol, rtol = create_init_inputs(
        N, D, ctx_num, his_num, tgt_num, batch_size, dtype
    )
    npu_out = fa_infer_with_arbitrary_func_fused(q_b, k_b, v_b, arbitrary_func, q2k_block_sparse)
    npu_out_cpu = npu_out.to("cpu")
    cpu_out = compute_reference_arbitrary_cpu(q_b, k_b, v_b, arbitrary_func, dtype == torch.float16)
    assert torch.allclose(cpu_out, npu_out_cpu, atol=atol, rtol=rtol)


@pytest.mark.graph
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("ctx_num", [0])
@pytest.mark.parametrize("his_num", [1000])
@pytest.mark.parametrize("tgt_num", [3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1])
def test_fa_infer_with_arbitrary_func_graph(N, D, ctx_num, his_num, tgt_num, batch_size, dtype):
    torch.npu.set_device(DEVICE)
    q_b, k_b, v_b, arbitrary_func, q2k_block_sparse, atol, rtol = create_init_inputs(
        N, D, ctx_num, his_num, tgt_num, batch_size, dtype
    )
    # 预处理与搬数在编译图外完成，仅编译算子调用本体
    q_npu = q_b.to(DEVICE)
    k_npu = k_b.to(DEVICE)
    v_npu = v_b.to(DEVICE)
    af_npu = arbitrary_func[:, :, 1:5, :].permute(0, 1, 3, 2).contiguous().to(DEVICE)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.fa_infer_with_arbitrary_func, dynamic=True, fullgraph=True)
    npu_out = compiled_fn(q_npu, k_npu, v_npu, af_npu, q2k_block_sparse)

    npu_out_cpu = npu_out.to("cpu")
    cpu_out = compute_reference_arbitrary_cpu(q_b, k_b, v_b, arbitrary_func, dtype == torch.float16)
    assert torch.allclose(cpu_out, npu_out_cpu, atol=atol, rtol=rtol)
