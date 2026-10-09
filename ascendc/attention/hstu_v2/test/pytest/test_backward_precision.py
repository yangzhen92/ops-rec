"""Parameterized HSTU V2 backward precision for the three supported call chains."""

import os
from dataclasses import replace

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    METADATA_ARBITRARY,
    NO_MASK,
    PLAN,
    PIPELINES,
    assert_matrix_coverage,
    assert_ragged_coverage,
    create_backends,
    precision_cases,
    pytest_params,
    ragged_precision_cases,
    run_precision_case,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
RAGGED_CASES = ragged_precision_cases()
CASES = precision_cases()
PARAMS = pytest_params(CASES, "bwd-precision-")
RAGGED_PARAMS = pytest_params(RAGGED_CASES, "bwd-ragged-")


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


def _assert_case(case, pipeline, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case("backward", case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail


@pytest.mark.parametrize("case", PARAMS)
def test_attn_meta_arbitrary_kernel(case, backends):
    _assert_case(case, METADATA_ARBITRARY, backends)


@pytest.mark.parametrize("case", PARAMS)
def test_arbitrary_kernel(case, backends):
    _assert_case(case, ARBITRARY, backends)


@pytest.mark.parametrize("case", PARAMS)
def test_builtin_hstu_mask(case, backends):
    _assert_case(case, BUILTIN, backends)


@pytest.mark.parametrize("case", PARAMS)
def test_no_mask(case, backends):
    _assert_case(case, NO_MASK, backends)


@pytest.mark.parametrize("case", PARAMS)
def test_plan(case, backends):
    _assert_case(case, PLAN, backends)


@pytest.mark.parametrize("case", RAGGED_PARAMS)
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_ragged_sequences(case, pipeline, backends):
    _assert_case(case, pipeline, backends)


CAUSAL_VARIANTS = (
    pytest.param(128, 0, 16, id="context"),
    pytest.param(0, 128, 16, id="target"),
    pytest.param(128, 128, 16, id="context-target"),
    pytest.param(64, 192, 32, id="mixed-boundary"),
)
CAUSAL_BASES = pytest_params((CASES[3], CASES[4], CASES[5], CASES[6]), "bwd-causal-")


@pytest.mark.parametrize("case", CAUSAL_BASES)
@pytest.mark.parametrize("num_context,num_target,target_group_size", CAUSAL_VARIANTS)
def test_builtin_hstu_context_target(case, num_context, num_target, target_group_size, backends):
    min_seq_len_q = min(case.seq_lens) if case.seq_lens is not None else case.max_seq_len
    case = replace(
        case,
        num_context=min(num_context, min_seq_len_q // 2),
        num_target=min(num_target, min_seq_len_q // 2),
        target_group_size=target_group_size,
    )
    _assert_case(case, BUILTIN, backends)


def test_get_bwd_tile_size():
    expected = {32: [256, 128, 128], 64: [256, 128, 128], 128: [256, 128, 128], 256: [128, 64, 256]}
    for dim, tile in expected.items():
        q = torch.empty((128, 1, dim))
        k = torch.empty((128, 1, dim))
        assert torch.ops.ops_rec.get_bwd_tile_size(q, k, q, False) == tile


def test_backward_precision_matrix_coverage():
    assert_matrix_coverage(CASES)
    assert_ragged_coverage(RAGGED_CASES)
    assert len(PARAMS) * len(PIPELINES) == 150
    assert len(RAGGED_PARAMS) * len(PIPELINES) == 85
    assert sum(case.max_seq_len > 2048 for case in CASES) == 6
