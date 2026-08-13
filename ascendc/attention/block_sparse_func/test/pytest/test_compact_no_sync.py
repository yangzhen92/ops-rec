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
"""Unit tests for ops_rec.ascendc.attention.compact_no_sync.

The NPU op is the sync-free replacement for flash-attn's ``compact_block_idx``.
It takes per-row counts of "full" and "masked" blocks plus a padded
``block_idx`` lookup table, and returns the CSR-style offsets and compacted
index buffers required by ``LinearBlockSparseTensors`` — without the
``cudaStreamSynchronize`` + ``.item()`` round-trips the CSR wrappers incur.

These tests build the expected output on the CPU with numpy/torch and
compare against the NPU op for shape, dtype, and value correctness.
"""

import logging

import sys
import pytest
import torch
import numpy as np

import ops_rec

torch.npu.config.allow_internal_format = False

# 测试有 144 种 shape 组合，增大 cache 避免 dynamic=True 时因重编译超限报错
torch._dynamo.config.cache_size_limit = 256

logging.basicConfig(level=logging.INFO)


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


# ---------------------------------------------------------------------------
# CPU reference implementation
# ---------------------------------------------------------------------------
def compact_no_sync_cpu(mask_cnt: np.ndarray, full_cnt: np.ndarray, block_idx: np.ndarray):
    """NumPy reference matching the NPU op.

    Returns a 6-tuple in the same order as the NPU op:
        (mask_cnt, mask_offset, mask_compact,
         full_cnt, full_offset, full_compact)

    The compact buffers are returned as length-``sum(count)`` arrays
    (i.e. only the valid prefix — the NPU op over-allocates the tail
    with ``torch::empty`` and we don't want to compare uninitialized data).
    """
    assert mask_cnt.shape == full_cnt.shape
    assert mask_cnt.ndim == 3
    assert block_idx.ndim == 4
    assert mask_cnt.shape[:3] == block_idx.shape[:3]
    B, H, num_blocks = mask_cnt.shape
    max_blocks = block_idx.shape[3]
    n_flat = B * H * num_blocks

    flat_mask = mask_cnt.reshape(-1)
    flat_full = full_cnt.reshape(-1)
    flat_bidx = block_idx.reshape(n_flat, max_blocks)

    mask_offset = np.zeros(n_flat + 1, dtype=np.int32)
    full_offset = np.zeros(n_flat + 1, dtype=np.int32)
    for i in range(n_flat):
        mask_offset[i + 1] = mask_offset[i] + int(flat_mask[i])
        full_offset[i + 1] = full_offset[i] + int(flat_full[i])

    total_mask = int(mask_offset[-1])
    total_full = int(full_offset[-1])

    mask_compact = np.zeros(total_mask, dtype=np.int32)
    full_compact = np.zeros(total_full, dtype=np.int32)

    for i in range(n_flat):
        mcnt = int(flat_mask[i])
        fcnt = int(flat_full[i])
        row = flat_bidx[i]
        for j in range(fcnt):
            full_compact[full_offset[i] + j] = row[j]
        for j in range(mcnt):
            mask_compact[mask_offset[i] + j] = row[max_blocks - 1 - j]

    return (
        mask_cnt.astype(np.int32, copy=True),
        mask_offset,
        mask_compact,
        full_cnt.astype(np.int32, copy=True),
        full_offset,
        full_compact,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _to_cpu_int(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy().astype(np.int32, copy=False)


def _check_tuple(npu_out, cpu_out, name: str = "compact_no_sync") -> None:
    """Assert all six return tensors match between NPU and CPU."""
    assert len(npu_out) == 6, f"[{name}] expected 6-tuple, got {len(npu_out)}"

    npu_mc, npu_mo, npu_mcp, npu_fc, npu_fo, npu_fcp = (t.cpu() for t in npu_out)
    cpu_mc, cpu_mo, cpu_mcp, cpu_fc, cpu_fo, cpu_fcp = cpu_out

    # Dtype checks
    for tag, t in (("mc", npu_mc), ("mo", npu_mo), ("mcp", npu_mcp), ("fc", npu_fc), ("fo", npu_fo), ("fcp", npu_fcp)):
        assert t.dtype == torch.int32, f"[{name}] {tag} dtype {t.dtype} != int32"

    # Shape checks for offsets
    assert npu_mo.shape == cpu_mo.shape, (
        f"[{name}] mask_offset shape mismatch: npu={tuple(npu_mo.shape)}, cpu={tuple(cpu_mo.shape)}"
    )
    assert npu_fo.shape == cpu_fo.shape, (
        f"[{name}] full_offset shape mismatch: npu={tuple(npu_fo.shape)}, cpu={tuple(cpu_fo.shape)}"
    )

    # Value checks: counts and offsets must match exactly
    assert torch.equal(npu_mc, torch.from_numpy(cpu_mc)), f"[{name}] mask_block_cnt mismatch"
    assert torch.equal(npu_fc, torch.from_numpy(cpu_fc)), f"[{name}] full_block_cnt mismatch"
    assert torch.equal(npu_mo, torch.from_numpy(cpu_mo)), f"[{name}] mask_offset mismatch"
    assert torch.equal(npu_fo, torch.from_numpy(cpu_fo)), f"[{name}] full_offset mismatch"

    # Compacts: the NPU op over-allocates the tail, so only check the valid prefix
    assert npu_mcp.shape[0] >= cpu_mcp.shape[0], (
        f"[{name}] mask_compact too small: npu={npu_mcp.shape[0]}, need>={cpu_mcp.shape[0]}"
    )
    assert npu_fcp.shape[0] >= cpu_fcp.shape[0], (
        f"[{name}] full_compact too small: npu={npu_fcp.shape[0]}, need>={cpu_fcp.shape[0]}"
    )
    assert torch.equal(npu_mcp[: cpu_mcp.shape[0]].to(torch.int32), torch.from_numpy(cpu_mcp)), (
        f"[{name}] mask_compact mismatch"
    )
    assert torch.equal(npu_fcp[: cpu_fcp.shape[0]].to(torch.int32), torch.from_numpy(cpu_fcp)), (
        f"[{name}] full_compact mismatch"
    )


# ---------------------------------------------------------------------------
# Fixture: skip everything if no NPU is available
# ---------------------------------------------------------------------------
def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU device not available")


# ---------------------------------------------------------------------------
# Hand-crafted small-case correctness
# ---------------------------------------------------------------------------
@pytest.mark.ci
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_hand_crafted_single_row(device):
    """A 1x1x1 row with fcnt=2, mcnt=1, max_blocks=4.

    block_idx[0,0,0] = [10, 20, 30, 40]

    Expected:
        full_offset = [0, 2], full_compact = [10, 20]
        mask_offset = [0, 1], mask_compact = [40]   (last mcnt=1 entry)
    """
    _require_npu()

    mask_cnt = np.array([[[1]]], dtype=np.int32)
    full_cnt = np.array([[[2]]], dtype=np.int32)
    block_idx = np.array([[[[10, 20, 30, 40]]]], dtype=np.int32)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "hand_crafted_single_row")

    # Spot-check the exact values for this case
    npu_mo, npu_fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    npu_mcp, npu_fcp = _to_cpu_int(npu_out[2]), _to_cpu_int(npu_out[5])
    assert npu_mo.tolist() == [0, 1]
    assert npu_fo.tolist() == [0, 2]
    # Compacts are over-allocated; slice to the valid prefix from the offset tail.
    assert npu_mcp[: int(npu_mo[-1])].tolist() == [40]
    assert npu_fcp[: int(npu_fo[-1])].tolist() == [10, 20]


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_hand_crafted_multi_row(device):
    """2x1x2 rows with mixed counts.

    block_idx[0,0,0] = [ 0,  1,  2,  3,  4,  5],  fcnt=2, mcnt=3
    block_idx[0,0,1] = [10, 11, 12, 13, 14, 15],  fcnt=4, mcnt=1
    block_idx[1,0,0] = [20, 21, 22, 23, 24, 25],  fcnt=0, mcnt=0
    block_idx[1,0,1] = [30, 31, 32, 33, 34, 35],  fcnt=6, mcnt=0
    """
    _require_npu()

    mask_cnt = np.array([[[3, 1]], [[0, 0]]], dtype=np.int32)
    full_cnt = np.array([[[2, 4]], [[0, 6]]], dtype=np.int32)
    block_idx = np.array(
        [[[[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]]], [[[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]]]],
        dtype=np.int32,
    )
    # B=2, H=1, num_blocks=2, max_blocks=6

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "hand_crafted_multi_row")

    # Spot-check
    npu_mo, npu_fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    npu_mcp, npu_fcp = _to_cpu_int(npu_out[2]), _to_cpu_int(npu_out[5])
    assert npu_mo.tolist() == [0, 3, 4, 4, 4]  # mask: 3, 1, 0, 0
    assert npu_fo.tolist() == [0, 2, 6, 6, 12]  # full: 2, 4, 0, 6
    # mask_compact picks the last mcnt entries of each row in REVERSED order
    # (matches the convention used by the all_masked test below):
    #   row 0 last 3 of [0,1,2,3,4,5] reversed -> [5,4,3]
    #   row 1 last 1 of [10..15]      reversed -> [15]
    #   row 2 nothing
    #   row 3 nothing
    assert npu_mcp[: int(npu_mo[-1])].tolist() == [5, 4, 3, 15]
    # full_compact picks first fcnt entries of each row:
    #   row 0 first 2 -> [0, 1]
    #   row 1 first 4 -> [10, 11, 12, 13]
    #   row 2 nothing
    #   row 3 first 6 -> [30, 31, 32, 33, 34, 35]
    assert npu_fcp[: int(npu_fo[-1])].tolist() == [0, 1, 10, 11, 12, 13, 30, 31, 32, 33, 34, 35]


# ---------------------------------------------------------------------------
# Parametric random tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize(
    "B,H,num_blocks,max_blocks",
    [
        (1, 1, 1, 4),
        (1, 1, 4, 4),
        (1, 2, 3, 5),
        (2, 1, 2, 3),
        (2, 3, 4, 8),
        (1, 1, 8, 16),
        (3, 2, 5, 7),
    ],
)
def test_compact_no_sync_random(device, B, H, num_blocks, max_blocks):
    """Random counts in [0, max_blocks] and random block_idx; compare to CPU."""
    _require_npu()

    rng = np.random.default_rng(seed=B * 1000 + H * 100 + num_blocks * 10 + max_blocks)

    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    # Some rows may have mask_cnt + full_cnt > max_blocks, but that is allowed
    # by the op's contract — only the prefix up to fcnt and suffix of size mcnt
    # are read.
    block_idx = rng.integers(0, 1000, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, f"random B={B} H={H} num={num_blocks} max={max_blocks}")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_all_zeros(device):
    """mask_cnt = full_cnt = 0 for every row.

    Expected: offsets are all zero (cumsum of zeros), compacts are empty.
    """
    _require_npu()

    B, H, num_blocks, max_blocks = 2, 3, 4, 5
    mask_cnt = np.zeros((B, H, num_blocks), dtype=np.int32)
    full_cnt = np.zeros((B, H, num_blocks), dtype=np.int32)
    block_idx = np.zeros((B, H, num_blocks, max_blocks), dtype=np.int32)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "all_zeros")

    npu_mo, npu_fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    assert np.all(npu_mo == 0)
    assert np.all(npu_fo == 0)


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_all_full(device):
    """Every row is fully populated: mask_cnt=0, full_cnt=max_blocks.

    Expected: mask_compact empty, full_compact = all rows' first max_blocks entries
    concatenated.
    """
    _require_npu()

    B, H, num_blocks, max_blocks = 1, 1, 3, 4
    mask_cnt = np.zeros((B, H, num_blocks), dtype=np.int32)
    full_cnt = np.full((B, H, num_blocks), max_blocks, dtype=np.int32)
    block_idx = np.arange(B * H * num_blocks * max_blocks, dtype=np.int32).reshape(B, H, num_blocks, max_blocks)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "all_full")

    npu_fcp = _to_cpu_int(npu_out[5])
    assert npu_fcp.tolist() == list(range(B * H * num_blocks * max_blocks))


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_all_masked(device):
    """Every row is fully masked: mask_cnt=max_blocks, full_cnt=0.

    Expected: full_compact empty, mask_compact = all rows' last max_blocks entries
    (in reverse order) concatenated.
    """
    _require_npu()

    B, H, num_blocks, max_blocks = 1, 1, 3, 4
    mask_cnt = np.full((B, H, num_blocks), max_blocks, dtype=np.int32)
    full_cnt = np.zeros((B, H, num_blocks), dtype=np.int32)
    block_idx = np.arange(B * H * num_blocks * max_blocks, dtype=np.int32).reshape(B, H, num_blocks, max_blocks)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "all_masked")

    npu_mcp = _to_cpu_int(npu_out[2])
    # For each row, the last max_blocks entries reversed = the row in reverse.
    # rows: [0,1,2,3]  reversed: [3,2,1,0]
    #       [4,5,6,7]  reversed: [7,6,5,4]
    #       [8,9,10,11] reversed: [11,10,9,8]
    assert npu_mcp.tolist() == [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_max_blocks_one(device):
    """max_blocks=1: each row has at most one full or masked block.

    The full/masked count must sum to ≤ 1 (else the suffix overlaps the prefix);
    the CPU reference allows larger sums, but the op's contract is that only
    the first fcnt and the last mcnt are read. With max_blocks=1 the first
    and last slots are the same, so we set fcnt+mcnt=1.
    """
    _require_npu()

    B, H, num_blocks = 2, 2, 4
    max_blocks = 1
    rng = np.random.default_rng(seed=42)
    # Each row gets either fcnt=1, mcnt=0 OR fcnt=0, mcnt=1
    full_cnt = rng.integers(0, 2, size=(B, H, num_blocks), dtype=np.int32)
    mask_cnt = 1 - full_cnt
    block_idx = rng.integers(0, 100, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "max_blocks_one")


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_output_metadata(device):
    """Return-tuple structure, dtypes, and shapes are as documented."""
    _require_npu()

    B, H, num_blocks, max_blocks = 2, 3, 5, 7
    rng = np.random.default_rng(seed=0)
    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    block_idx = rng.integers(0, 100, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )

    assert len(npu_out) == 6
    n_flat = B * H * num_blocks

    mc, mo, mcp, fc, fo, fcp = npu_out
    assert mc.shape == (B, H, num_blocks)
    assert fc.shape == (B, H, num_blocks)
    assert mo.shape == (n_flat + 1,)
    assert fo.shape == (n_flat + 1,)
    assert mcp.shape == (n_flat * max_blocks,)
    assert fcp.shape == (n_flat * max_blocks,)

    for tag, t in (("mc", mc), ("mo", mo), ("mcp", mcp), ("fc", fc), ("fo", fo), ("fcp", fcp)):
        assert t.dtype == torch.int32, f"{tag} dtype {t.dtype} != int32"
        assert t.device.type == "npu", f"{tag} not on NPU"


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_offset_starts_at_zero(device):
    """The first entry of each offset array must be zero (exclusive scan)."""
    _require_npu()

    B, H, num_blocks, max_blocks = 1, 1, 4, 4
    rng = np.random.default_rng(seed=7)
    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    block_idx = rng.integers(0, 100, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )

    mo, fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    assert mo[0] == 0
    assert fo[0] == 0


@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_offset_total_equals_sum(device):
    """The final offset entry equals the sum of all per-row counts."""
    _require_npu()

    B, H, num_blocks, max_blocks = 2, 2, 3, 5
    rng = np.random.default_rng(seed=11)
    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    block_idx = rng.integers(0, 100, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    npu_out = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )

    mo, fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    assert mo[-1] == int(mask_cnt.sum())
    assert fo[-1] == int(full_cnt.sum())


# ---------------------------------------------------------------------------
# Reproducibility & determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_deterministic(device):
    """Calling the op twice on the same inputs must yield the same outputs."""
    _require_npu()

    B, H, num_blocks, max_blocks = 2, 2, 4, 6
    rng = np.random.default_rng(seed=2026)
    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    block_idx = rng.integers(0, 1000, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    npu_out_1 = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )
    npu_out_2 = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx).to(device),
    )

    for tag, t1, t2 in zip(("mc", "mo", "mcp", "fc", "fo", "fcp"), npu_out_1, npu_out_2):
        if tag in ("mcp", "fcp"):
            # Only compare valid prefix (the tail is uninitialized)
            n1 = int(_to_cpu_int(npu_out_1[1])[-1]) if tag == "mcp" else int(_to_cpu_int(npu_out_1[4])[-1])
            assert torch.equal(t1[:n1].to(torch.int32), t2[:n1].to(torch.int32)), (
                f"[deterministic] {tag} differs between calls"
            )
        else:
            assert torch.equal(t1, t2), f"[deterministic] {tag} differs between calls"


# ---------------------------------------------------------------------------
# Mask/Full ordering independence: the two streams are independent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_full_and_mask_independent(device):
    """The mask stream and the full stream are independent: changes to one
    do not affect the other.
    """
    _require_npu()

    block_idx_a = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.int32)
    block_idx_b = block_idx_a.copy()

    # Same full_cnt, varying mask_cnt: the full stream must not change
    full_cnt = np.array([[[2, 3]]], dtype=np.int32)
    mask_cnt_a = np.array([[[0, 0]]], dtype=np.int32)
    mask_cnt_b = np.array([[[2, 3]]], dtype=np.int32)

    out_a = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt_a).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx_a).to(device),
    )
    out_b = ops_rec.ascendc.attention.compact_no_sync(
        torch.from_numpy(mask_cnt_b).to(device),
        torch.from_numpy(full_cnt).to(device),
        torch.from_numpy(block_idx_b).to(device),
    )

    # full stream is identical between the two calls
    assert torch.equal(out_a[4], out_b[4])  # full_offset
    n_full = int(_to_cpu_int(out_a[4])[-1])
    assert torch.equal(out_a[5][:n_full].to(torch.int32), out_b[5][:n_full].to(torch.int32))

    # sanity: mask stream DID change — otherwise the assertion above is trivial
    assert not torch.equal(out_a[1], out_b[1])  # mask_offset


@pytest.mark.graph
@pytest.mark.parametrize("device", ["npu:0"])
def test_compact_no_sync_graph(device):
    _require_npu()

    mask_cnt = np.array([[[1]]], dtype=np.int32)
    full_cnt = np.array([[[2]]], dtype=np.int32)
    block_idx = np.array([[[[10, 20, 30, 40]]]], dtype=np.int32)

    cpu_out = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    compiled_fn = torch_compiled(ops_rec.ascendc.attention.compact_no_sync, dynamic=True, fullgraph=True)
    npu_out = compiled_fn(
        mask_block_cnt=torch.from_numpy(mask_cnt).to(device),
        full_block_cnt=torch.from_numpy(full_cnt).to(device),
        block_idx=torch.from_numpy(block_idx).to(device),
    )
    _check_tuple(npu_out, cpu_out, "hand_crafted_single_row")

    # Spot-check the exact values for this case
    npu_mo, npu_fo = _to_cpu_int(npu_out[1]), _to_cpu_int(npu_out[4])
    npu_mcp, npu_fcp = _to_cpu_int(npu_out[2]), _to_cpu_int(npu_out[5])
    assert npu_mo.tolist() == [0, 1]
    assert npu_fo.tolist() == [0, 2]
    # Compacts are over-allocated; slice to the valid prefix from the offset tail.
    assert npu_mcp[: int(npu_mo[-1])].tolist() == [40]
    assert npu_fcp[: int(npu_fo[-1])].tolist() == [10, 20]


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
