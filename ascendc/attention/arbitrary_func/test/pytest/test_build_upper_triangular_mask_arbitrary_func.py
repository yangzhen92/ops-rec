"""Independent correctness and packed integration tests for upper_triangular."""

import random
import numpy as np
import pytest
import torch
import ops_rec  # noqa: F401

DEVICE = "npu:0"
N_FUNC = 2


def make_case(index):
    rng = random.Random(20260921 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (17, 32, 65, 129, 257)[index % 5]
    lengths = [rng.randint(max(2, cap // 3), cap) for _ in range(batch)]
    lengths[0] = cap
    lengths[-1] = max(2, cap // 2)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    config = {
        "lengths": lengths,
        "offsets": offsets,
    }
    return config


def expected_dense(config):
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for batch, length in enumerate(config["lengths"]):
        start = config["offsets"][batch]
        for query in range(length):
            for key in range(length):
                dense[start + query, start + key] = key >= query
    return dense


def expected_intervals(config):
    dense = expected_dense(config)
    total = dense.shape[0]
    encoded = np.zeros((1, 1, total, N_FUNC), dtype=np.int32)
    for batch, length in enumerate(config["lengths"]):
        start, end = config["offsets"][batch : batch + 2]
        for query in range(start, end):
            active = np.flatnonzero(dense[query, start:end]) + start
            intervals = []
            if active.size:
                run_start = run_end = int(active[0])
                for key in active[1:]:
                    key = int(key)
                    if key != run_end + 1:
                        intervals.append((run_start, run_end + 1))
                        run_start = key
                    run_end = key
                intervals.append((run_start, run_end + 1))
            if not intervals:
                intervals.append((start, start))
            assert len(intervals) == 1
            encoded[0, 0, query] = np.asarray(intervals, dtype=np.int32).reshape(-1)
    return encoded


def run_operator(config):
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)

    total = config["offsets"][-1]
    function = torch.ops.ops_rec.build_upper_triangular_mask_arbitrary_func
    return function(offsets, total), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_upper_triangular_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("direction", ("q2k", "k2q"))
def test_upper_triangular_packed(direction):
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
