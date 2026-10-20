"""Regression coverage for metadata direction, batch inference and validation."""

import os

import pytest
import torch
import ops_rec  # noqa: F401 - registers operators and packaged OPP paths

from test_create_attn_metadata import assert_all_row_blocks_covered


@pytest.fixture(autouse=True)
def select_device():
    torch.npu.set_device(int(os.getenv("ASCEND_DEVICE_ID", "0")))


def metadata(*, lengths=(257, 65), layout_q="TND", layout_kv="TND", row_axis=1, packed=False, **overrides):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int32, device="npu")
    used = torch.tensor(lengths, dtype=torch.int32, device="npu")
    outer_block = 64 if row_axis == 0 else 128
    counts = torch.ones(
        (len(lengths), 1, (max(lengths) + outer_block - 1) // outer_block), dtype=torch.int32, device="npu"
    )
    args = dict(
        cu_seqlens_q=cu if layout_q == "TND" else None,
        cu_seqlens_kv=cu if layout_kv == "TND" else None,
        seqused_q=None if layout_q == "TND" else used,
        seqused_kv=None if layout_kv == "TND" else used,
        batch_size=-1,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        num_heads_q=2,
        num_heads_kv=2,
        head_dim=64,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out="TND",
        mask_block_cnt=counts if packed else None,
        full_block_cnt=counts if packed else None,
        packed_row_axis=row_axis,
        packed_mask_heads=1,
        q_block_size=64,
        kv_block_size=128,
        deterministic=True,
    )
    args.update(overrides)
    return torch.ops.ops_rec.create_attn_metadata(**args)


@pytest.mark.parametrize("layout_q,layout_kv", [("TND", "BSND"), ("BSND", "TND"), ("BNSD", "BSND")])
def test_backward_mixed_layout_matches_tnd(layout_q, layout_kv):
    actual = metadata(layout_q=layout_q, layout_kv=layout_kv).cpu()
    expected = metadata().cpu()
    assert tuple(actual[2:4].tolist()) == (128, 64)
    words = 16 + int(expected[0]) * int(expected[14]) * 16
    torch.testing.assert_close(actual[:words], expected[:words], rtol=0, atol=0)
    assert_all_row_blocks_covered(actual, (257, 65), 2)


@pytest.mark.parametrize("row_axis", [0, 1])
def test_packed_independent_seqused_lengths(row_axis):
    actual = metadata(layout_q="BSND", layout_kv="BSND", row_axis=row_axis, packed=True).cpu()
    assert int(actual[15]) == 0x48535455
    assert_all_row_blocks_covered(actual, (257, 65), 2)


def test_gqa_rejected_before_launch():
    with pytest.raises(RuntimeError, match="MHA"):
        metadata(num_heads_q=4)


@pytest.mark.parametrize("mask_mode", [0, 3])
def test_non_band_window_rejected_before_launch(mask_mode):
    with pytest.raises(RuntimeError, match="win_left|winLeft"):
        metadata(mask_mode=mask_mode, win_left=10)


def test_missing_batch_inference_rejected_before_launch():
    with pytest.raises(RuntimeError, match="batch_size"):
        metadata(cu_seqlens_q=None, seqused_q=None, layout_q="BSND")
