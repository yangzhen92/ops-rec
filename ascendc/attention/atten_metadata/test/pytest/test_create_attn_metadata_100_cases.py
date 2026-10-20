"""Task2: deterministic 100-case HSTU metadata validation matrix."""

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "benchmarks/hstu_v2_agent_task2"))
from cases import operator_cases  # noqa: E402
from test_create_attn_metadata import assert_all_row_blocks_covered  # noqa: E402

METADATA_STRIDE = 16
DIRECT_MAGIC_INDEX = 15
HSTU_DIRECT_MAGIC = 0x48535455
AIC_CORE_INDEX = 14


def _build_inputs(case):
    offsets = torch.tensor(case["offsets"], dtype=torch.int32, device="npu")
    zeros = torch.zeros(case["batch"], dtype=torch.int32, device="npu")
    arbitrary = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(offsets, zeros, zeros, 1, case["offsets"][-1])
    q_block = 128
    packed = torch.ops.ops_rec.create_fwd_packed_block_sparse(
        arbitrary, offsets, offsets, case["max_seq_len"], case["max_seq_len"], q_block, 128
    )
    return offsets, packed


@pytest.mark.parametrize("case", operator_cases(), ids=lambda c: f"case{c['case_id']:03d}")
def test_create_attn_metadata_100_cases(case):
    torch.npu.set_device(0)
    offsets = torch.tensor(case["offsets"], dtype=torch.int32, device="npu")
    mask_cnt = full_cnt = None
    if case["has_arbitrary"]:
        offsets, packed = _build_inputs(case)
        mask_cnt, full_cnt = packed[0], packed[3]

    metadata = torch.ops.ops_rec.create_attn_metadata(
        offsets,
        offsets,
        None,
        None,
        case["batch"],
        case["max_seq_len"],
        case["max_seq_len"],
        case["heads"],
        case["heads"],
        case["head_dim"],
        0,
        -1,
        -1,
        "TND",
        "TND",
        "TND",
        mask_cnt,
        full_cnt,
        0,
        1,
    ).cpu()

    assert metadata.dtype == torch.int32
    aic_cores = int(metadata[AIC_CORE_INDEX])
    assert aic_cores > 0
    assert_all_row_blocks_covered(metadata, case["seq_lens"], case["heads"])
    if case["has_arbitrary"]:
        assert int(metadata[DIRECT_MAGIC_INDEX]) == HSTU_DIRECT_MAGIC
        assert int(metadata[0]) == 1
    else:
        assert int(metadata[0]) >= 1
    assert int(metadata[2]) > 0
    assert int(metadata[3]) > 0
