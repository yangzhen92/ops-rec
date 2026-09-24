"""Independent correctness and packed integration tests for target_group_block."""

import random
import numpy as np
import pytest
import torch
import ops_rec  # noqa: F401

DEVICE = "npu:0"
N_FUNC = 4


def make_case(index):
    rng = random.Random(20260921 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (17, 32, 65, 129, 257)[index % 5]
    lengths = [rng.randint(max(2, cap // 3), cap) for _ in range(batch)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    config = {
        "lengths": lengths,
        "offsets": offsets,
        "target_lens": [min(length, batch + 1 + (index % 3)) for batch, length in enumerate(lengths)],
        "group_sizes": [min(length, batch + 2 + (index % 5)) for batch, length in enumerate(lengths)],
    }
    if batch > 1:
        assert len(set(config["target_lens"])) > 1
    if batch > 1:
        assert len(set(config["group_sizes"])) > 1
    return config


def expected_dense(config):
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for batch, length in enumerate(config["lengths"]):
        start = config["offsets"][batch]
        for query in range(length):
            for key in range(length):
                dense[start + query, start + key] = (
                    key <= query < length - config["target_lens"][batch]
                ) or (
                    query >= length - config["target_lens"][batch]
                    and (
                        key < length - config["target_lens"][batch]
                        or (
                            key >= length - config["target_lens"][batch]
                            and (key - (length - config["target_lens"][batch])) // config["group_sizes"][batch]
                            == (query - (length - config["target_lens"][batch])) // config["group_sizes"][batch]
                        )
                    )
                )
    return dense


def expected_intervals(config):
    total = config["offsets"][-1]
    encoded = np.zeros((1, 1, total, N_FUNC), dtype=np.int32)
    for batch, length in enumerate(config["lengths"]):
        start, end = config["offsets"][batch : batch + 2]
        target = config["target_lens"][batch]
        history = length - target
        group = config["group_sizes"][batch]
        for local_query in range(length):
            query = start + local_query
            if local_query < history:
                end0 = query + 1
                encoded[0, 0, query] = (start, end0, end0, end0)
            else:
                group_start = (local_query - history) // group * group
                begin1 = start + history + group_start
                end1 = start + history + min(group_start + group, target)
                encoded[0, 0, query] = (start, start + history, begin1, end1)
    return encoded


def run_operator(config):
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)
    tensors = {
        key: torch.tensor(config[key], dtype=torch.int32, device=DEVICE) for key in ["target_lens", "group_sizes"]
    }
    total = config["offsets"][-1]
    function = torch.ops.ops_rec.build_target_group_block_mask_arbitrary_func
    return function(offsets, tensors["target_lens"], tensors["group_sizes"], total), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_target_group_block_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("direction", ("q2k", "k2q"))
def test_target_group_block_packed(direction):
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
