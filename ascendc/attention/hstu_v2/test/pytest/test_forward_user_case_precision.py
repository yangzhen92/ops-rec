"""Forward precision validation of the official user_case matrix (builtin causal path)."""

import os

import pytest
import torch

from utils.feature_matrix import (
    BUILTIN,
    create_backends,
    dense_reference_elements,
    run_precision_case,
    user_case_precision_cases,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
CASES = user_case_precision_cases()
# The PyTorch golden materializes dense [B, H, Nq, Nk] fp32 score matrices plus
# several same-shape intermediates; shapes beyond this budget are skipped
# explicitly instead of exhausting host memory.  Currently unlimited.
REFERENCE_ELEMENT_BUDGET = float("inf")
EXPECTED_SKIPPED = 0


def _user_case_id(case):
    return (
        f"uc-fwd-{case.case_id:03d}-b{case.batch_size}-h{case.head_num}-"
        f"q{case.max_seq_len}-k{case.resolved_max_seq_len_k}-"
        f"d{case.head_dim}-dv{case.resolved_head_dim_v}-{case.dtype_name}-rab{int(case.has_rab)}"
    )


def _params():
    selected = []
    for case in CASES:
        marks = []
        if dense_reference_elements(case) > REFERENCE_ELEMENT_BUDGET:
            marks = [pytest.mark.skip(reason="dense fp32 golden exceeds memory budget")]
        selected.append(pytest.param(case, marks=marks, id=_user_case_id(case)))
    return selected


PARAMS = _params()


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("case", PARAMS)
def test_user_case_forward_precision(case, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case("forward", case, BUILTIN, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail


def test_user_case_forward_matrix():
    skipped = sum(dense_reference_elements(case) > REFERENCE_ELEMENT_BUDGET for case in CASES)
    assert skipped == EXPECTED_SKIPPED
