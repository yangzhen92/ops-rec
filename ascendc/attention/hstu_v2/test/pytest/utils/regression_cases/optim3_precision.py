"""Precision and parameter gaps identified by the optim3 design guide."""

import os

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    METADATA_ARBITRARY,
    NO_MASK,
    TestCaseParams,
    _make_case_data,
    _run_fused,
    create_backends,
    run_precision_case,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
TestCaseParams.__test__ = False


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


GEOMETRY_CASES = (
    TestCaseParams(301, 2, 2, 128, 96, False, torch.float16, 2026092801, 0, 0, 1, (64, 128)),
    TestCaseParams(302, 3, 1, 100, 192, True, torch.bfloat16, 2026092802, 0, 0, 1, (65, 100, 33)),
    TestCaseParams(303, 2, 2, 128, 96, False, torch.bfloat16, 2026092803, 16, 16, 4, (64, 128)),
    TestCaseParams(304, 2, 1, 100, 192, False, torch.float16, 2026092804, 16, 16, 8, (100, 33)),
    TestCaseParams(307, 3, 2, 100, 64, False, torch.float16, 2026092807, 0, 0, 1, (33, 65, 100)),
)


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("pipeline", (METADATA_ARBITRARY, ARBITRARY, BUILTIN, NO_MASK))
@pytest.mark.parametrize("case", GEOMETRY_CASES, ids=lambda case: f"case{case.case_id}-d{case.head_dim}")
def test_non_aligned_sequences_and_intermediate_dims(direction, pipeline, case, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case(direction, case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail


SCALE_ALPHA_CASE = TestCaseParams(305, 2, 2, 128, 96, False, torch.float16, 2026092805, 16, 16, 4, (64, 128))
SCALE_ALPHA_CONFIGS = (
    pytest.param(0.5, 0.0, id="alpha0.5-default-scale"),
    pytest.param(1.0, 1 / 128, id="alpha1-explicit-default-scale"),
    pytest.param(2.0, 1 / 777, id="alpha2-custom-scale"),
)


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("pipeline", (METADATA_ARBITRARY, ARBITRARY, BUILTIN, NO_MASK))
@pytest.mark.parametrize("alpha,scale", SCALE_ALPHA_CONFIGS)
def test_alpha_and_scale_matrix(direction, pipeline, alpha, scale, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case(
        direction, SCALE_ALPHA_CASE, pipeline, fused, native, validator, f"npu:{DEVICE_ID}", alpha=alpha, scale=scale
    )
    assert passed, detail


@pytest.mark.parametrize("pipeline", (METADATA_ARBITRARY, ARBITRARY, BUILTIN))
def test_masked_rab_gradient_is_exactly_zero(pipeline, backends):
    case = TestCaseParams(306, 2, 2, 100, 64, True, torch.float16, 2026092806, 16, 16, 4, (65, 100))
    data = _make_case_data(case)
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    fused = backends[0].kernel(0.5, 1 / 100, True, 100, 100, offsets_q, offsets_k)
    result = _run_fused("backward", case, pipeline, fused, data, f"npu:{DEVICE_ID}")
    rab_grad = result[3]
    assert rab_grad is not None
    assert torch.count_nonzero(rab_grad[dense_mask == 0]).item() == 0


def _small_inputs(heads=1):
    length, dim = 33, 32
    q = torch.randn((length, heads, dim), dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    grad = torch.randn_like(q)
    offsets = [0, length]
    return grad, q, k, v, offsets


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("alpha", (0.0, -0.5))
def test_non_positive_alpha_rejected(direction, alpha, backends):
    grad, q, k, v, offsets = _small_inputs()
    kernel = backends[0].kernel(alpha, 1 / 33, False, 33, 33, offsets, offsets)
    with pytest.raises(RuntimeError, match="alpha must be greater than 0"):
        if direction == "forward":
            kernel.forward(q, k, v, None, None)
        else:
            kernel.backward(grad, q, k, v)


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_more_than_sixteen_heads_rejected(direction, backends):
    grad, q, k, v, offsets = _small_inputs(heads=17)
    kernel = backends[0].kernel(0.5, 1 / 33, False, 33, 33, offsets, offsets)
    with pytest.raises(RuntimeError, match=r"heads must be in \[1,16\]"):
        if direction == "forward":
            kernel.forward(q, k, v, None, None)
        else:
            kernel.backward(grad, q, k, v)
