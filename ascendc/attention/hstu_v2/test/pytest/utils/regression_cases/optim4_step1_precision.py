"""Guide2 step1 coverage that is not present in the legacy HSTU matrix."""

import os

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    METADATA_ARBITRARY,
    NO_MASK,
    TestCaseParams,
    create_backends,
    run_precision_case,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
TestCaseParams.__test__ = False


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


SHARED_RAB_CASES = (
    TestCaseParams(401, 1, 4, 128, 64, True, torch.float16, 2026092901, 0, 0, 1, (128,), 1),
    TestCaseParams(402, 2, 4, 100, 96, True, torch.bfloat16, 2026092902, 0, 0, 1, (65, 100), 1),
)


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("pipeline", (NO_MASK, BUILTIN, ARBITRARY, METADATA_ARBITRARY))
@pytest.mark.parametrize("case", SHARED_RAB_CASES, ids=lambda item: f"case{item.case_id}")
def test_shared_rab_heads(direction, pipeline, case, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case(direction, case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail


def test_invalid_rab_head_count_rejected(backends):
    case = SHARED_RAB_CASES[0]
    from utils.feature_matrix import _make_case_data

    _, q, k, v, rab, _, offsets_q, offsets_k = _make_case_data(case)
    invalid_rab = rab.repeat(1, 2, 1, 1)
    kernel = backends[0].kernel(
        0.5, 1 / case.max_seq_len, True, case.max_seq_len, case.max_seq_len, offsets_q, offsets_k
    )
    with pytest.raises(RuntimeError, match="rab shape mismatch"):
        kernel.forward(q, k, v, invalid_rab, None)


def test_shared_rab_with_nondefault_scale(backends):
    case = SHARED_RAB_CASES[1]
    fused, native, validator = backends
    passed, detail = run_precision_case(
        "forward",
        case,
        BUILTIN,
        fused,
        native,
        validator,
        f"npu:{DEVICE_ID}",
        alpha=0.37,
        scale=0.003125,
    )
    assert passed, detail
