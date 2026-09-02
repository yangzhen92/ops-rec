#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Unit tests for ops_rec.ascendc.attention.create_q2k_block_sparse_from_func
and ops_rec.ascendc.attention.create_k2q_block_sparse_from_func.

Migrated from ``test_create_block_mask_npu.py`` (RecSDK dynamic_emb).  Both
the Q2K (forward) and K2Q (backward) ops are exercised.

The NPU ops convert a function-encoded arbitrary mask (``func_tensor``) into
a block-sparse representation following the FlexAttention block-mask
convention.  This migration uses the NEW ``func_tensor`` layout
``[B, H, func_q_len, n_func]`` where ``n_func`` is even and the valid kv
intervals for a query token are ``[F0, F1)`` ∪ ``[F2, F3)`` ∪ ...

* For each (q_block, kv_block) pair the block is classified as:
    * ``FULL``  — every (q_token × kv_position) inside the block is valid
    * ``MASK``  — some, but not all, positions are valid
    * ``EMPTY`` — no positions are valid
* Q2K (forward) returns ``(mask_block_cnt, full_block_cnt, block_idx)`` shaped
  ``[B, H, num_q_blocks]`` / ``[B, H, num_q_blocks, num_kv_blocks]``.  Inside
  each row of ``block_idx`` the ``full`` kv-block ids are written
  left-to-right and the ``mask`` ids right-to-left (holes are ``-1``).
* K2Q (backward) is the transpose: shapes ``[B, H, num_kv_blocks]`` /
  ``[B, H, num_kv_blocks, num_q_blocks]``.  K2Q always performs boundary
  checking (a partial q_block can never be FULL).

Reference values are produced by a block-level CPU reference that mirrors the
kernel's FULL/MASK/EMPTY classification without materialising the full
``B * H * Q_LEN * KV_LEN`` boolean mask.  Counts must match exactly; per row
the kernel and reference index sets must be equal.
"""

import logging

import sys
import pytest
import torch

import ops_rec

torch.npu.config.allow_internal_format = False
logging.basicConfig(level=logging.INFO)

# 测试有 144 种 shape 组合，增大 cache 避免 dynamic=True 时因重编译超限报错
torch._dynamo.config.cache_size_limit = 256

CPU_DEVICE = "cpu"
NPU_DEVICE = "npu:0"


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


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

                state_full = block_full.all(dim=1)  # [K]
                state_any = (block_full | block_mask).any(dim=1)  # [K]

                if is_q_block_full:
                    state_mask = state_any & ~state_full  # [K]
                else:
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


def create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=None):
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = create_reference_block_mask(
        func_tensor,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        device,
    )

    bm_B, bm_H, num_q_blocks = ref_mask_cnt.shape
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    if device is None:
        device = ref_mask_cnt.device

    mask_block_cnt = torch.zeros(bm_B, bm_H, num_kv_blocks, dtype=torch.int32, device=device)
    full_block_cnt = torch.zeros(bm_B, bm_H, num_kv_blocks, dtype=torch.int32, device=device)
    mask_block_idx = torch.full((bm_B, bm_H, num_kv_blocks, num_q_blocks), -1, dtype=torch.int32, device=device)
    full_block_idx = torch.full((bm_B, bm_H, num_kv_blocks, num_q_blocks), -1, dtype=torch.int32, device=device)

    for b in range(bm_B):
        for h in range(bm_H):
            for q_block in range(num_q_blocks):
                fcnt = int(ref_full_cnt[b, h, q_block].item())
                for i in range(fcnt):
                    kv_block = int(ref_full_idx[b, h, q_block, i].item())
                    pos = int(full_block_cnt[b, h, kv_block].item())
                    full_block_idx[b, h, kv_block, pos] = q_block
                    full_block_cnt[b, h, kv_block] = pos + 1

                mcnt = int(ref_mask_cnt[b, h, q_block].item())
                for i in range(mcnt):
                    kv_block = int(ref_mask_idx[b, h, q_block, i].item())
                    pos = int(mask_block_cnt[b, h, kv_block].item())
                    mask_block_idx[b, h, kv_block, pos] = q_block
                    mask_block_cnt[b, h, kv_block] = pos + 1

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def extract_indices_from_combined(mask_cnt, full_cnt, block_idx):
    """Split the combined ``block_idx`` into separate full / mask index tensors.

    Layout: full block ids fill positions ``[0, full_cnt)`` left-to-right;
    mask block ids fill positions ``[max_blocks - mask_cnt, max_blocks)``
    right-to-left, so we ``flip(0)`` them before comparing.
    """
    B, H, num_blocks, max_blocks = block_idx.shape
    mask_idx = torch.full_like(block_idx, -1)
    full_idx = torch.full_like(block_idx, -1)

    for b in range(B):
        for h in range(H):
            for blk in range(num_blocks):
                fcnt = int(full_cnt[b, h, blk].item())
                mcnt = int(mask_cnt[b, h, blk].item())
                if fcnt > 0:
                    full_idx[b, h, blk, :fcnt] = block_idx[b, h, blk, :fcnt]
                if mcnt > 0:
                    tail = block_idx[b, h, blk, max_blocks - mcnt : max_blocks]
                    mask_idx[b, h, blk, :mcnt] = tail.flip(0)
    return mask_idx, full_idx


def create_q2k_kernel_block_mask(
    func_tensor_npu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True, debug=True, graph=False
):
    if graph:
        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.create_q2k_block_sparse_from_func, dynamic=True, fullgraph=True
        )
        mask_block_cnt, full_block_cnt, block_idx = compiled_fn(
            func_tensor=func_tensor_npu,
            q_len=seqlen_q,
            kv_len=seqlen_k,
            q_block_size=Q_BLOCK_SIZE,
            kv_block_size=KV_BLOCK_SIZE,
            check_q_boundary=check_q_boundary,
            debug=debug,
        )
    else:
        mask_block_cnt, full_block_cnt, block_idx = ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor_npu,
            seqlen_q,
            seqlen_k,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            check_q_boundary,
            debug,
        )

    mask_block_cnt = mask_block_cnt.cpu()
    full_block_cnt = full_block_cnt.cpu()
    block_idx = block_idx.cpu()

    mask_block_idx, full_block_idx = extract_indices_from_combined(mask_block_cnt, full_block_cnt, block_idx)
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx


def create_k2q_kernel_block_mask(
    func_tensor_npu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, debug=True, graph=False
):
    if graph:
        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.create_k2q_block_sparse_from_func, dynamic=True, fullgraph=True
        )
        mask_block_cnt, full_block_cnt, block_idx = compiled_fn(
            func_tensor=func_tensor_npu,
            q_len=seqlen_q,
            kv_len=seqlen_k,
            q_block_size=Q_BLOCK_SIZE,
            kv_block_size=KV_BLOCK_SIZE,
            debug=debug,
        )
    else:
        mask_block_cnt, full_block_cnt, block_idx = ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor_npu,
            seqlen_q,
            seqlen_k,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            debug,
        )

    mask_block_cnt = mask_block_cnt.cpu()
    full_block_cnt = full_block_cnt.cpu()
    block_idx = block_idx.cpu()

    mask_block_idx, full_block_idx = extract_indices_from_combined(mask_block_cnt, full_block_cnt, block_idx)
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx


def compare_block_masks(ref_cnt, ref_idx, kernel_cnt, kernel_idx, name=""):
    if not torch.equal(ref_cnt, kernel_cnt):
        print(f"{name} count mismatch!")
        print(f"  Reference: {ref_cnt}")
        print(f"  Kernel:    {kernel_cnt}")
        return False

    B, H, num_blocks = ref_cnt.shape
    for b in range(B):
        for h in range(H):
            for blk in range(num_blocks):
                cnt = int(ref_cnt[b, h, blk].item())
                if cnt == 0:
                    continue
                ref_set = set(int(x) for x in ref_idx[b, h, blk, :cnt].tolist())
                ker_set = set(int(x) for x in kernel_idx[b, h, blk, :cnt].tolist())
                if ref_set != ker_set:
                    print(f"{name} index mismatch at (b={b}, h={h}, blk={blk})!")
                    print(f"  Reference: {sorted(ref_set)}")
                    print(f"  Kernel:    {sorted(ker_set)}")
                    return False
    return True


# ---------------------------------------------------------------------------
# func_tensor generators
# ---------------------------------------------------------------------------
def _zeros_cpu(B, H, n_func, seqlen_q):
    """Allocate the int32 func_tensor on CPU device."""
    return torch.zeros(B, H, n_func, seqlen_q, dtype=torch.int32, device=CPU_DEVICE)


def generate_func_tensor(seqlen_q, seqlen_k, n_func=3, seed=0, B=1, H=1):
    g = torch.Generator(device=CPU_DEVICE).manual_seed(seed)
    func_tensor = _zeros_cpu(B, H, n_func, seqlen_q)
    coef = 1.0 / n_func
    for i in range(n_func):
        if i == 0:
            # actual: F0 = 0 (skip [0, 0))
            func_tensor[:, :, 0, :seqlen_q] = 0
        else:
            low = int(i * coef * seqlen_k)
            high = int((i + 1) * coef * seqlen_k)
            if high <= low:
                high = low + 1
            func_tensor[:, :, i, :seqlen_q] = torch.randint(
                low,
                high,
                size=(B, H, seqlen_q),
                dtype=torch.int32,
                generator=g,
            )
    return func_tensor.contiguous()


def generate_special_func_tensor(seqlen_q, seqlen_k, pattern, n_func=3, block_size=128, B=1, H=1):
    func_tensor = _zeros_cpu(B, H, n_func, seqlen_q)
    q_indices = torch.arange(seqlen_q, dtype=torch.int32, device=CPU_DEVICE)

    if pattern == "empty":
        # All positions invalid: F0 = 0 → first interval [0, 0) is empty.
        func_tensor[:, :, 0, :seqlen_q] = 0

    elif pattern == "full":
        # All positions valid: F0 = seqlen_k → first interval [0, seqlen_k).
        func_tensor[:, :, 0, :seqlen_q] = seqlen_k

    elif pattern == "causal":
        # Causal: q_idx attends to [0, q_idx + 1).
        func_tensor[:, :, 0, :seqlen_q] = torch.minimum(
            q_indices + 1,
            torch.full_like(q_indices, seqlen_k),
        )

    elif pattern == "anti_causal":
        # Anti-causal: q_idx attends to [q_idx, seqlen_k).
        # Encoded as F0 = 0 (skip [0, 0)), F1 = q_idx, F2 = seqlen_k.
        func_tensor[:, :, 0, :seqlen_q] = 0
        func_tensor[:, :, 1, :seqlen_q] = torch.minimum(
            q_indices,
            torch.full_like(q_indices, seqlen_k),
        )
        func_tensor[:, :, 2, :seqlen_q] = seqlen_k

    elif pattern == "diagonal":
        # Block-level diagonal: each q_block only attends to the corresponding
        # kv_block. Needs n_func >= 3.
        block_start = (q_indices // block_size) * block_size
        block_end = torch.minimum(
            block_start + block_size,
            torch.full_like(block_start, seqlen_k),
        )
        func_tensor[:, :, 0, :seqlen_q] = 0
        func_tensor[:, :, 1, :seqlen_q] = block_start
        func_tensor[:, :, 2, :seqlen_q] = block_end

    elif pattern == "sliding_window":
        window_size = 2 * block_size
        window_start = torch.maximum(
            q_indices - window_size // 2,
            torch.zeros_like(q_indices),
        )
        window_end = torch.minimum(
            q_indices + window_size // 2,
            torch.full_like(q_indices, seqlen_k),
        )
        func_tensor[:, :, 0, :seqlen_q] = 0
        func_tensor[:, :, 1, :seqlen_q] = window_start
        func_tensor[:, :, 2, :seqlen_q] = window_end

    elif pattern == "checkerboard":
        # Alternating block pattern: even q_blocks see even kv_blocks, etc.
        for q_idx in range(seqlen_q):
            blk_q = q_idx // block_size
            if blk_q % 2 == 0:
                func_tensor[0, 0, 0, q_idx] = min(block_size, seqlen_k)
                if n_func >= 3:
                    func_tensor[0, 0, 1, q_idx] = min(2 * block_size, seqlen_k)
                    func_tensor[0, 0, 2, q_idx] = min(3 * block_size, seqlen_k)
            else:
                func_tensor[0, 0, 0, q_idx] = 0
                if n_func >= 3:
                    func_tensor[0, 0, 1, q_idx] = min(block_size, seqlen_k)
                    func_tensor[0, 0, 2, q_idx] = min(2 * block_size, seqlen_k)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    # actual: F0 = 0 (skip [0, 0))
    func_tensor[:, :, 0, :seqlen_q] = 0
    return func_tensor.contiguous()


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU device not available")


def _to_npu(func_tensor_cpu):
    return func_tensor_cpu.to(NPU_DEVICE)


def _check_q2k(
    func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True, tag="", graph=False
):
    torch.npu.set_device(NPU_DEVICE)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = create_reference_block_mask(
        func_tensor_cpu,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    # reference: [B, H, n_func, totalseqlen] -> NPU: [B, H, totalseqlen, n_func]
    func_tensor_cpu = func_tensor_cpu[:, :, 1 : func_tensor_cpu.size(2), :].permute(0, 1, 3, 2).contiguous()
    ker_mask_cnt, ker_mask_idx, ker_full_cnt, ker_full_idx, _ = create_q2k_kernel_block_mask(
        _to_npu(func_tensor_cpu),
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=check_q_boundary,
        debug=False,
        graph=graph,
    )

    mask_ok = compare_block_masks(
        ref_mask_cnt,
        ref_mask_idx,
        ker_mask_cnt,
        ker_mask_idx,
        f"[{tag}] mask_block",
    )
    full_ok = compare_block_masks(
        ref_full_cnt,
        ref_full_idx,
        ker_full_cnt,
        ker_full_idx,
        f"[{tag}] full_block",
    )
    assert mask_ok, f"[{tag}] mask block mismatch"
    assert full_ok, f"[{tag}] full block mismatch"


def _check_k2q(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag="", graph=False):
    torch.npu.set_device(NPU_DEVICE)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = create_reference_k2q_block_mask(
        func_tensor_cpu,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    func_tensor_cpu = func_tensor_cpu[:, :, 1 : func_tensor_cpu.size(2), :].permute(0, 1, 3, 2).contiguous()
    ker_mask_cnt, ker_mask_idx, ker_full_cnt, ker_full_idx, _ = create_k2q_kernel_block_mask(
        _to_npu(func_tensor_cpu),
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        debug=True,
        graph=graph,
    )

    mask_ok = compare_block_masks(
        ref_mask_cnt,
        ref_mask_idx,
        ker_mask_cnt,
        ker_mask_idx,
        f"[{tag}] mask_block",
    )
    full_ok = compare_block_masks(
        ref_full_cnt,
        ref_full_idx,
        ker_full_cnt,
        ker_full_idx,
        f"[{tag}] full_block",
    )
    assert mask_ok, f"[{tag}] mask block mismatch"
    assert full_ok, f"[{tag}] full block mismatch"


@pytest.mark.ci
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (4096, 4096),
        (4111, 4111),
        (8192, 8192),
        (16384, 16384),
        (16666, 16666),
        (32768, 32768),
        (33333, 33333),
        (61111, 61111),
    ],
)
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize(
    "Q_BLOCK_SIZE,KV_BLOCK_SIZE",
    [
        (128, 128),
    ],
)
def test_q2k_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Q2K random arbitrary-mask patterns."""
    _require_npu()
    seed = seqlen_q * 100 + n_func * 10 + Q_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, seed=seed)
    tag = f"random(seqlen={seqlen_q}, n_func={n_func}, Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})"
    _check_q2k(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True, tag=tag)


# ---------------------------------------------------------------------------
# 1b. K2Q random-mask correctness sweep (migrated from test_k2q_random_mask)
# ---------------------------------------------------------------------------
@pytest.mark.ci
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (4096, 4096),
        (4111, 4111),
        (8192, 8192),
        (16384, 16384),
        (16666, 16666),
        (32768, 32768),
        (33333, 33333),
        (61111, 61111),
    ],
)
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize(
    "Q_BLOCK_SIZE,KV_BLOCK_SIZE",
    [
        (128, 128),
    ],
)
def test_k2q_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """K2Q random arbitrary-mask patterns (backward view)."""
    _require_npu()
    seed = seqlen_q * 100 + n_func * 10 + Q_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, seed=seed)
    tag = f"k2q_random(seqlen={seqlen_q}, n_func={n_func}, Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})"
    _check_k2q(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=tag)


# ---------------------------------------------------------------------------
# 2. Special attention patterns (migrated from test_special_patterns_q2k)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pattern",
    [
        "empty",
        "full",
        "causal",
        "anti_causal",
        "diagonal",
        "sliding_window",
    ],
)
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 4096, 4111])
def test_special_patterns_q2k(pattern, seqlen):
    """Q2K against hand-crafted attention patterns."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 3 if pattern in ("anti_causal", "diagonal", "sliding_window", "checkerboard") else 1
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        pattern,
        n_func=n_func,
        block_size=KV_BLOCK_SIZE,
    )
    _check_q2k(
        func_tensor_cpu,
        seqlen,
        seqlen,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"special({pattern}, seqlen={seqlen})",
    )


# ---------------------------------------------------------------------------
# 2b. Special attention patterns — K2Q (backward) view
# (migrated from test_special_patterns_k2q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pattern",
    [
        "empty",
        "full",
        "causal",
        "anti_causal",
        "diagonal",
        "sliding_window",
    ],
)
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 4096, 4111])
def test_special_patterns_k2q(pattern, seqlen):
    """K2Q against hand-crafted attention patterns (backward view)."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 3 if pattern in ("anti_causal", "diagonal", "sliding_window", "checkerboard") else 1
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        pattern,
        n_func=n_func,
        block_size=KV_BLOCK_SIZE,
    )
    _check_k2q(
        func_tensor_cpu, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=f"k2q_special({pattern}, seqlen={seqlen})"
    )


# ---------------------------------------------------------------------------
# 3. Small-seqlen boundary tests (migrated from test_small_seqlen_q2k)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seqlen",
    [
        1,
        2,
        7,
        15,
        31,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
    ],
)
def test_small_seqlen_q2k(seqlen):
    """Sub-block and just-over-block sequence lengths."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        "causal",
        n_func=1,
    )
    _check_q2k(
        func_tensor_cpu,
        seqlen,
        seqlen,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"small_seqlen({seqlen})",
    )


# ---------------------------------------------------------------------------
# 3b. Small-seqlen boundary tests — K2Q (backward) view
# (migrated from test_small_seqlen_k2q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seqlen",
    [
        1,
        2,
        7,
        15,
        31,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
    ],
)
def test_small_seqlen_k2q(seqlen):
    """K2Q sub-block and just-over-block sequence lengths."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        "causal",
        n_func=1,
    )
    _check_k2q(func_tensor_cpu, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=f"k2q_small_seqlen({seqlen})")


# ---------------------------------------------------------------------------
# 4. Asymmetric Q/K length tests (migrated from test_asymmetric_seqlen_q2k)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 256),  # Q shorter than K
        (256, 128),  # Q longer than K
        (100, 300),  # non-aligned, Q < K
        (300, 100),  # non-aligned, Q > K
        (1, 1000),  # single Q token
        (1000, 1),  # single K token
        (127, 129),  # just under/over block boundary
        (129, 127),
        (255, 257),  # 2x block boundary
        (257, 255),
    ],
)
def test_asymmetric_seqlen_q2k(seqlen_q, seqlen_k):
    """Independent Q/K sequence lengths, including degenerate single-token cases."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    func_tensor_cpu = generate_special_func_tensor(
        seqlen_q,
        seqlen_k,
        "full",
        n_func=1,
    )
    _check_q2k(
        func_tensor_cpu,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"asym(Q={seqlen_q}, K={seqlen_k})",
    )


# ---------------------------------------------------------------------------
# 4b. Asymmetric Q/K length tests — K2Q (backward) view
# (migrated from test_asymmetric_seqlen_k2q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 256),  # Q shorter than K
        (256, 128),  # Q longer than K
        (100, 300),  # non-aligned, Q < K
        (300, 100),  # non-aligned, Q > K
        (1, 1000),  # single Q token
        (1000, 1),  # single K token
        (127, 129),  # just under/over block boundary
        (129, 127),
    ],
)
def test_asymmetric_seqlen_k2q(seqlen_q, seqlen_k):
    """K2Q independent Q/K sequence lengths, including degenerate single-token cases."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    func_tensor_cpu = generate_special_func_tensor(
        seqlen_q,
        seqlen_k,
        "full",
        n_func=1,
    )
    _check_k2q(
        func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=f"k2q_asym(Q={seqlen_q}, K={seqlen_k})"
    )


# ---------------------------------------------------------------------------
# 5. Block-size sweep (migrated from test_different_block_sizes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [(128, 128)])
def test_different_block_sizes(Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Q and KV block sizes can be chosen independently."""
    _require_npu()
    seqlen = 1024
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        "causal",
        n_func=1,
    )
    _check_q2k(
        func_tensor_cpu,
        seqlen,
        seqlen,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"block_sizes(Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})",
    )


# ---------------------------------------------------------------------------
# 6. Grid-overflow tests: total_blocks > 65535
#
# In ``launch_create_q2k_block_sparse_from_func`` the 1D grid is clamped to
# 65535 whenever ``total_blocks = num_q_blocks * H * B`` reaches that limit:
#
#     int grid = total_blocks >= 65535 ? 65535 : total_blocks;
#
# These cases pick (B, H, seqlen) so the product crosses the threshold and
# exercise that branch. The (B, H) grid is built up rather than the seqlen
# so the per-test memory cost stays modest (~150-300 MB for the func_tensor
# + block_idx outputs).
#
# ``check_q_boundary=True`` is preserved so the launcher keeps its default
# behaviour. The pattern is ``"full"`` because the FlexAttention reference
# computes a single (b, h) mask and reuses it across the (B, H) grid; using
# a uniform pattern (same func for every (b, h) pair) keeps the kernel and
# reference in agreement.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B,H,seqlen",
    [
        # # num_q_blocks = seqlen / 128 = 256,  total = 256 * 16 * 16 = 65536
        # (16, 16, 32768),
        # # num_q_blocks = seqlen / 128 = 512,  total = 512 *  8 * 16 = 65536
        # (8, 16, 65536),
        # # num_q_blocks = seqlen / 128 = 512,  total = 512 *  4 * 32 = 65536
        (4, 32, 65536),
    ],
)
def test_q2k_grid_overflow(B, H, seqlen):
    """``total_blocks >= 65535``: launcher's grid-clamp branch is exercised.

    The kernel is launched with a 1D grid of 65535 even though more
    (b, h, q_block) tuples exist; the kernel is expected to cover every
    tuple (typically via a stride loop). The ``"full"`` pattern keeps the
    per-(b, h) result dense (full_block_cnt == num_kv_blocks) so a missed
    tuple shows up immediately as a count mismatch.
    """
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    num_q_blocks = (seqlen + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    total_blocks = num_q_blocks * H * B
    # Sanity-check the parametrize: every row must trigger the clamp.
    assert total_blocks >= 65535, (
        f"Test misconfigured: total_blocks={total_blocks} does not cross "
        f"the 65535 grid-clamp threshold for (B={B}, H={H}, seqlen={seqlen})."
    )

    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        "full",
        n_func=1,
        B=B,
        H=H,
    )
    _check_q2k(
        func_tensor_cpu,
        seqlen,
        seqlen,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"grid_overflow(B={B},H={H},seqlen={seqlen},total={total_blocks})",
    )


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (32832, 32832),  # 8*4072+256
    ],
)
@pytest.mark.parametrize("n_func", [5])
@pytest.mark.parametrize(
    "Q_BLOCK_SIZE,KV_BLOCK_SIZE",
    [
        (128, 128),
    ],
)
def test_q2k_random_mask_customer(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Q2K random arbitrary-mask patterns."""
    _require_npu()
    seed = seqlen_q * 100 + n_func * 10 + Q_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, B=1, H=1, seed=seed)

    tag = f"random(seqlen={seqlen_q}, n_func={n_func}, Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})"
    _check_q2k(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True, tag=tag)


@pytest.mark.graph
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (32832, 32832),  # 8*4072+256
    ],
)
@pytest.mark.parametrize("n_func", [5])
@pytest.mark.parametrize(
    "Q_BLOCK_SIZE,KV_BLOCK_SIZE",
    [
        (128, 128),
    ],
)
def test_q2k_random_mask_customer_graph(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Q2K random arbitrary-mask patterns."""
    _require_npu()
    seed = seqlen_q * 100 + n_func * 10 + Q_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, B=1, H=1, seed=seed)

    tag = f"random(seqlen={seqlen_q}, n_func={n_func}, Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})"
    _check_q2k(
        func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True, tag=tag, graph=True
    )


@pytest.mark.graph
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (32832, 32832),  # 8*4072+256
    ],
)
@pytest.mark.parametrize("n_func", [5])
@pytest.mark.parametrize(
    "Q_BLOCK_SIZE,KV_BLOCK_SIZE",
    [
        (128, 128),
    ],
)
def test_k2q_random_mask_customer_graph(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """K2Q random arbitrary-mask patterns."""
    _require_npu()
    seed = seqlen_q * 100 + n_func * 10 + Q_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, B=1, H=1, seed=seed)

    tag = f"random(seqlen={seqlen_q}, n_func={n_func}, Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})"
    _check_k2q(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=tag, graph=True)


# ---------------------------------------------------------------------------
# 7. Block-size alignment argument validation
#
# Both the Q2K and K2Q entry points in block_sparse_func_plugin.cpp enforce
# (catlass restriction):
#
#     TORCH_CHECK(q_block_size > 0 && q_block_size % 32 == 0,
#                 "q_block_size must be positive and divisible by 32");
#     TORCH_CHECK(kv_block_size > 0 && kv_block_size % 32 == 0,
#                 "kv_block_size must be positive and divisible by 32");
#
# The tests below verify that a non-aligned (or non-positive) block size is
# rejected by both Q2K and K2Q with the matching message (legal block sizes
# are covered by test_different_block_sizes above).
# ---------------------------------------------------------------------------
Q_BLOCK_SIZE_ERR_MSG = "q_block_size must be positive and divisible by 32"
KV_BLOCK_SIZE_ERR_MSG = "kv_block_size must be positive and divisible by 32"

# 非 32 的倍数或非正数——每一个取值都必须被拒绝
INVALID_BLOCK_SIZES = [
    31,  # 最小对齐尺寸减一
    33,  # 32 加一
    48,  # 非 32 的倍数
    100,  # 非 32 的倍数
    0,  # 非正数
    -32,  # 负数（虽为 32 的倍数）
    -1,  # 负数
]


def _minimal_npu_func_tensor(seqlen_q=64, n_func=2, B=1, H=1):
    """构造最小规模的 NPU 布局 func_tensor [B, H, func_q_len, n_func]（n_func 为偶数）。"""
    func_tensor = torch.zeros(B, H, seqlen_q, n_func, dtype=torch.int32, device=CPU_DEVICE)
    return _to_npu(func_tensor.contiguous())


@pytest.mark.parametrize("bad_q_block_size", INVALID_BLOCK_SIZES)
def test_invalid_q_block_size(bad_q_block_size):
    """Q2K and K2Q both reject a q_block_size that is not a positive multiple of 32."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor()
    # q2k: 非法 q_block_size（合法 kv_block_size）
    with pytest.raises(RuntimeError, match=Q_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=bad_q_block_size,
            kv_block_size=128,
            check_q_boundary=True,
            debug=False,
        )
    # k2q: 非法 q_block_size（合法 kv_block_size）
    with pytest.raises(RuntimeError, match=Q_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=bad_q_block_size,
            kv_block_size=128,
            debug=False,
        )


@pytest.mark.parametrize("bad_kv_block_size", INVALID_BLOCK_SIZES)
def test_invalid_kv_block_size(bad_kv_block_size):
    """Q2K and K2Q both reject a kv_block_size that is not a positive multiple of 32."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor()
    # q2k: 合法 q_block_size，非法 kv_block_size
    with pytest.raises(RuntimeError, match=KV_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=bad_kv_block_size,
            check_q_boundary=True,
            debug=False,
        )
    # k2q: 合法 q_block_size，非法 kv_block_size
    with pytest.raises(RuntimeError, match=KV_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=bad_kv_block_size,
            debug=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
