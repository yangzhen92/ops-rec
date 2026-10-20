import os
import pytest
import torch

import ops_rec


def direct_state(descriptor):
    return (descriptor.cpu().to(torch.int32) & 3).to(torch.int32)


def create_packed(direction, *args):
    op = (
        torch.ops.ops_rec.create_fwd_packed_block_sparse
        if direction == "q2k"
        else torch.ops.ops_rec.create_bwd_packed_block_sparse
    )
    return op(*args)


VALID_BLOCKS = (
    ("q2k", 32, 32),
    ("q2k", 32, 64),
    ("q2k", 32, 128),
    ("q2k", 64, 64),
    ("q2k", 64, 128),
    ("q2k", 128, 128),
    ("k2q", 32, 32),
    ("k2q", 64, 32),
    ("k2q", 64, 64),
    ("k2q", 128, 32),
    ("k2q", 128, 64),
    ("k2q", 128, 128),
    ("k2q", 256, 32),
    ("k2q", 256, 64),
    ("k2q", 256, 128),
)


@pytest.mark.parametrize("direction,q_block,kv_block", VALID_BLOCKS)
@pytest.mark.parametrize("length", [256, 1024, 4096])
def test_descriptor_matches_cpu_reference(direction, q_block, kv_block, length):
    torch.npu.set_device(int(os.getenv("ASCEND_DEVICE_ID", "0")))
    offsets = torch.tensor([0, length], dtype=torch.int32, device="npu")
    func = ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(
        offsets,
        torch.tensor([128], dtype=torch.int32, device="npu"),
        torch.tensor([128], dtype=torch.int32, device="npu"),
        16,
        length,
    )
    packed = create_packed(direction, func, offsets, offsets, length, length, q_block, kv_block)
    torch.npu.synchronize()

    func_cpu = func.cpu()
    packed_mask_cnt = packed[0].cpu()
    bounds = packed[1].cpu().reshape(-1, 2)
    packed_full_cnt = packed[3].cpu()
    num_q = (length + q_block - 1) // q_block
    num_k = (length + kv_block - 1) // kv_block
    inner_blocks = num_q if direction == "k2q" else num_k
    descriptor = direct_state(packed[6]).reshape(-1, inner_blocks)
    expected_mask_cnt = torch.zeros_like(packed_mask_cnt).reshape(-1)
    expected_full_cnt = torch.zeros_like(packed_full_cnt).reshape(-1)
    outer_tile = kv_block if direction == "k2q" else q_block
    outer_blocks = (length + outer_tile - 1) // outer_tile
    inner_blocks = descriptor.shape[1]
    for row in range(descriptor.shape[0]):
        expected = torch.zeros_like(descriptor[row])
        outer_block = row % outer_blocks
        for inner in range(inner_blocks):
            q_block_id, kv_block_id = (inner, outer_block) if direction == "k2q" else (outer_block, inner)
            q_begin = q_block_id * q_block
            q_end = min(q_begin + q_block, length)
            kv_begin = kv_block_id * kv_block
            kv_end = min(kv_begin + kv_block, length)
            all_visible = True
            any_visible = False
            for q_pos in range(q_begin, q_end):
                row_func = func_cpu[0, 0, q_pos]
                visible = torch.zeros(kv_end - kv_begin, dtype=torch.bool)
                for group in range(0, row_func.numel(), 2):
                    begin, end = map(int, row_func[group : group + 2])
                    left, right = max(begin, kv_begin), min(end, kv_end)
                    if left < right:
                        visible[left - kv_begin : right - kv_begin] = True
                any_visible |= bool(visible.any())
                all_visible &= bool(visible.all()) and kv_end - kv_begin == kv_block
            state = 1 if all_visible else (3 if any_visible else 0)
            expected[inner] = state
            expected_mask_cnt[row] += state == 3
            expected_full_cnt[row] += state == 1
        assert torch.equal(descriptor[row] & 3, expected)
        nonempty = torch.nonzero(expected, as_tuple=False).flatten()
        expected_bounds = [-1, -1] if nonempty.numel() == 0 else [int(nonempty[0]), int(nonempty[-1])]
        assert bounds[row].tolist() == expected_bounds
    assert torch.equal(packed_mask_cnt.reshape(-1), expected_mask_cnt)
    assert torch.equal(packed_full_cnt.reshape(-1), expected_full_cnt)

    assert packed[2].numel() == 1
    assert packed[7].numel() == 1


@pytest.mark.parametrize("direction,q_block", [("q2k", 128), ("k2q", 256)])
def test_default_keeps_placeholder_metadata(direction, q_block):
    torch.npu.set_device(int(os.getenv("ASCEND_DEVICE_ID", "0")))
    length = 256
    offsets = torch.tensor([0, length], dtype=torch.int32, device="npu")
    func = ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(
        offsets,
        torch.tensor([128], dtype=torch.int32, device="npu"),
        torch.tensor([128], dtype=torch.int32, device="npu"),
        16,
        length,
    )
    packed = create_packed(direction, func, offsets, offsets, length, length, q_block, 128)
    assert packed[2].numel() == 1


@pytest.mark.parametrize("direction,q_block", [("q2k", 128), ("k2q", 256)])
def test_varlen_tnd_multi_sequence(direction, q_block):
    """Global TND AF is partitioned by offsets without batch padding."""
    torch.npu.set_device(int(os.getenv("ASCEND_DEVICE_ID", "0")))
    seq_lens = [129, 385, 777]
    max_seqlen = max(seq_lens)
    offsets = torch.tensor([0, 129, 514, 1291], dtype=torch.int32, device="npu")
    zeros = torch.zeros(len(seq_lens), dtype=torch.int32, device="npu")
    func = ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(offsets, zeros, zeros, 1, sum(seq_lens))

    packed = create_packed(direction, func, offsets, offsets, max_seqlen, max_seqlen, q_block, 128)
    torch.npu.synchronize()

    num_q = (max_seqlen + q_block - 1) // q_block
    num_k = (max_seqlen + 127) // 128
    outer, inner = (num_k, num_q) if direction == "k2q" else (num_q, num_k)
    assert packed[0].shape == (len(seq_lens), 1, outer)
    assert packed[1].shape == (len(seq_lens), 1, outer, 2)
    assert packed[6].shape == (len(seq_lens), 1, outer, inner)
    assert packed[7].numel() == 1

    descriptors = direct_state(packed[6])
    bounds = packed[1].cpu()
    for batch_idx, seq_len in enumerate(seq_lens):
        actual_outer = (seq_len + (128 if direction == "k2q" else q_block) - 1) // (
            128 if direction == "k2q" else q_block
        )
        if actual_outer < outer:
            assert torch.equal(
                descriptors[batch_idx, 0, actual_outer:], torch.zeros_like(descriptors[batch_idx, 0, actual_outer:])
            )
            assert torch.equal(
                bounds[batch_idx, 0, actual_outer:], torch.full_like(bounds[batch_idx, 0, actual_outer:], -1)
            )
        assert (descriptors[batch_idx, 0, :actual_outer] != 0).any()
