"""Precision coverage for the repeated short-Q/long-K scheduler fast path."""

import os

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    NO_MASK,
    PLAN,
    TestCaseParams as CaseParams,
    create_backends,
    run_precision_case,
)


DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
CASES = (
    pytest.param(
        CaseParams(
            1001,
            8,
            4,
            52,
            64,
            False,
            torch.float16,
            2026100101,
            0,
            0,
            1,
            (52,) * 8,
            None,
            1000,
            64,
            (1000,) * 8,
        ),
        id="b8-h4-q52-k1000-d64",
    ),
    pytest.param(
        CaseParams(
            1002,
            16,
            4,
            32,
            64,
            False,
            torch.float16,
            2026100102,
            0,
            0,
            1,
            (32,) * 16,
            None,
            499,
            32,
            (499,) * 16,
        ),
        id="b16-h4-q32-k499-d64-v32",
    ),
)


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("pipeline", (NO_MASK, BUILTIN, ARBITRARY, PLAN))
@pytest.mark.parametrize("case", CASES)
def test_repeated_short_q_long_k_backward(case, pipeline, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case("backward", case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail
