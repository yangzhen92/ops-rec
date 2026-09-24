"""Task2: deterministic 100-case packed block-sparse CPU comparison."""

import pytest
import torch


from cases import operator_cases


def _expected_descriptor(arbitrary_cpu, case, direction, q_block, kv_block):
    max_len = case["max_seq_len"]
    num_q = (max_len + q_block - 1) // q_block
    num_k = (max_len + kv_block - 1) // kv_block
    outer, inner = (num_q, num_k) if direction == "q2k" else (num_k, num_q)
    expected = torch.zeros((case["batch"], 1, outer, inner), dtype=torch.int32)
    offset = 0
    for batch_idx, seq_len in enumerate(case["seq_lens"]):
        rows = arbitrary_cpu[0, 0, offset : offset + seq_len]
        visible = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        for q_pos in range(seq_len):
            for pair in range(0, rows.shape[-1], 2):
                start = max(offset, int(rows[q_pos, pair])) - offset
                end = min(offset + seq_len, int(rows[q_pos, pair + 1])) - offset
                if start < end:
                    visible[q_pos, start:end] = True
        actual_q_blocks = (seq_len + q_block - 1) // q_block
        actual_k_blocks = (seq_len + kv_block - 1) // kv_block
        for q_idx in range(actual_q_blocks):
            q_begin, q_end = q_idx * q_block, min((q_idx + 1) * q_block, seq_len)
            for k_idx in range(actual_k_blocks):
                k_begin, k_end = k_idx * kv_block, min((k_idx + 1) * kv_block, seq_len)
                block = visible[q_begin:q_end, k_begin:k_end]
                if not bool(block.any()):
                    continue
                is_full = q_end - q_begin == q_block and k_end - k_begin == kv_block and bool(block.all())
                outer_idx, inner_idx = (q_idx, k_idx) if direction == "q2k" else (k_idx, q_idx)
                expected[batch_idx, 0, outer_idx, inner_idx] = 1 if is_full else 3
        offset += seq_len
    return expected


def _direct_state(descriptor):
    return (descriptor.cpu().to(torch.int32) & 3).to(torch.int32)


def _create_packed(direction, *args):
    op = (
        torch.ops.ops_rec.create_fwd_packed_block_sparse
        if direction == "q2k"
        else torch.ops.ops_rec.create_bwd_packed_block_sparse
    )
    return op(*args)


@pytest.mark.parametrize("case", operator_cases(), ids=lambda c: f"case{c['case_id']:03d}")
def test_packed_block_sparse_100_cases(case):
    torch.npu.set_device(0)
    offsets = torch.tensor(case["offsets"], dtype=torch.int32, device="npu")
    zeros = torch.zeros(case["batch"], dtype=torch.int32, device="npu")
    arbitrary = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(offsets, zeros, zeros, 1, case["offsets"][-1])
    q_block = 128 if case["direction"] == "q2k" else 256
    packed = _create_packed(
        case["direction"], arbitrary, offsets, offsets, case["max_seq_len"], case["max_seq_len"], q_block, 128
    )
    torch.npu.synchronize()

    expected = _expected_descriptor(arbitrary.cpu(), case, case["direction"], q_block, 128)
    assert packed[6].shape == expected.shape
    assert packed[7].numel() == 1
    actual = _direct_state(packed[6])
    assert torch.equal(actual, expected)

    mask_cnt = (expected == 3).sum(dim=-1).to(torch.int32)
    full_cnt = (expected == 1).sum(dim=-1).to(torch.int32)
    assert torch.equal(packed[0].cpu(), mask_cnt)
    assert torch.equal(packed[3].cpu(), full_cnt)

    nonempty = expected != 0
    bounds = torch.full((*expected.shape[:-1], 2), -1, dtype=torch.int32)
    for index in torch.nonzero(nonempty.any(dim=-1), as_tuple=False):
        b, h, outer = map(int, index)
        positions = torch.nonzero(nonempty[b, h, outer], as_tuple=False).flatten()
        bounds[b, h, outer] = torch.tensor([int(positions[0]), int(positions[-1])])
    assert torch.equal(packed[1].cpu(), bounds)
