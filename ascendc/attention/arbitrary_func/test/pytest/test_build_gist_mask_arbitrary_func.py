"""Correctness and packed integration tests for gist-cache mask."""

import random
import numpy as np
import pytest
import torch
import ops_rec  # noqa: F401

DEVICE = "npu:0"
N_FUNC = 4


def make_case(index):
    rng = random.Random(20260921 + 5000 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (33, 65, 129, 257, 513)[index % 5]
    lengths = [rng.randint(max(16, cap // 2), cap) for _ in range(batch)]
    lengths[0] = cap
    offsets = [0]
    pre_doc_lens = []
    gist_starts = []
    gist_ends = []
    for batch_id, length in enumerate(lengths):
        pre_doc = min(3 + (index + batch_id) % 7, max(1, length // 4))
        gist_len = min(4 + (index + 2 * batch_id) % 11, max(1, length // 5))
        doc_min = pre_doc + 1
        gist_start = rng.randint(doc_min, max(doc_min, length - gist_len - 1))
        gist_end = min(length, gist_start + gist_len)
        if gist_end >= length:
            gist_end = length - 1
            gist_start = max(pre_doc + 1, gist_end - gist_len)
        pre_doc_lens.append(pre_doc)
        gist_starts.append(gist_start)
        gist_ends.append(gist_end)
        offsets.append(offsets[-1] + length)
    return {
        "lengths": lengths,
        "offsets": offsets,
        "pre_doc_lens": pre_doc_lens,
        "gist_starts": gist_starts,
        "gist_ends": gist_ends,
    }


def add_interval(intervals, begin, end, seq_start, seq_end):
    begin = max(begin, seq_start)
    end = min(end, seq_end)
    if begin >= end:
        return
    if intervals and intervals[-1][1] == begin:
        intervals[-1] = (intervals[-1][0], end)
    else:
        intervals.append((begin, end))


def expected_intervals(config):
    total = config["offsets"][-1]
    encoded = np.zeros((1, 1, total, N_FUNC), dtype=np.int32)
    for batch_id, length in enumerate(config["lengths"]):
        seq_start, seq_end = config["offsets"][batch_id : batch_id + 2]
        p = config["pre_doc_lens"][batch_id]
        g0 = config["gist_starts"][batch_id]
        g1 = config["gist_ends"][batch_id]
        for local_q in range(length):
            global_q = seq_start + local_q
            intervals = []
            if local_q < p:
                add_interval(intervals, seq_start, global_q + 1, seq_start, seq_end)
            elif local_q < g0:
                add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
                add_interval(intervals, seq_start + p, global_q + 1, seq_start, seq_end)
            elif local_q < g1:
                add_interval(intervals, seq_start, global_q + 1, seq_start, seq_end)
            else:
                add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
                add_interval(intervals, seq_start + g0, seq_start + g1, seq_start, seq_end)
                add_interval(intervals, seq_start + g1, global_q + 1, seq_start, seq_end)
            while len(intervals) < 2:
                last = intervals[-1][1] if intervals else seq_start
                intervals.append((last, last))
            assert len(intervals) <= 2
            encoded[0, 0, global_q] = np.asarray(intervals[:2], dtype=np.int32).reshape(-1)
    return encoded


def expected_dense(config):
    encoded = expected_intervals(config)
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for row in range(total):
        for begin, end in encoded[0, 0, row].reshape(-1, 2):
            if begin < end:
                dense[row, begin:end] = True
    return dense


def run_operator(config):
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)
    pre_doc_lens = torch.tensor(config["pre_doc_lens"], dtype=torch.int32, device=DEVICE)
    gist_starts = torch.tensor(config["gist_starts"], dtype=torch.int32, device=DEVICE)
    gist_ends = torch.tensor(config["gist_ends"], dtype=torch.int32, device=DEVICE)
    total = config["offsets"][-1]
    return torch.ops.ops_rec.build_gist_mask_arbitrary_func(
        offsets, pre_doc_lens, gist_starts, gist_ends, total
    ), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_gist_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("direction", ("q2k", "k2q"))
def test_gist_mask_packed(direction):
    config = make_case(29)
    encoded, offsets = run_operator(config)
    maximum = max(config["lengths"])
    q_block = 128 if direction == "q2k" else 256
    op = (
        torch.ops.ops_rec.create_fwd_packed_block_sparse
        if direction == "q2k"
        else torch.ops.ops_rec.create_bwd_packed_block_sparse
    )
    packed = op(encoded, offsets, offsets, maximum, maximum, q_block, 128)
    decoded = packed[6].cpu().to(torch.int32) & 3
    assert torch.all((decoded == 0) | (decoded == 1) | (decoded == 3))
    assert int((decoded == 3).sum()) == int(packed[0].sum().cpu())
    assert int((decoded == 1).sum()) == int(packed[3].sum().cpu())
