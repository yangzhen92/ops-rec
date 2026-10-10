"""Parameterized HSTU V2 backward precision: feature matrix, user cases, and graph mode."""

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
    _hstu_parameters,
    _make_case_data,
    assert_matrix_coverage,
    assert_ragged_coverage,
    create_backends,
    precision_cases,
    pytest_params,
    ragged_precision_cases,
    run_precision_case,
    user_case_precision_cases,
)
from utils.graph_mode import (
    ALPHA,
    GRAPH_CAUSAL_VARIANT_PARAMS,
    GRAPH_CONTEXT_VALUE,
    GRAPH_PARAMS,
    GRAPH_TARGET_GROUP_SIZE,
    MASK_TYPE_CAUSAL,
    SCALE,
    run_compiled,
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


# 官方 user_case 用例矩阵，走 builtin causal 路径。
USER_CASES = user_case_precision_cases()


def _user_case_id(case):
    return (
        f"uc-bwd-{case.case_id:03d}-b{case.batch_size}-h{case.head_num}-"
        f"q{case.max_seq_len}-k{case.resolved_max_seq_len_k}-"
        f"d{case.head_dim}-dv{case.resolved_head_dim_v}-{case.dtype_name}-rab{int(case.has_rab)}"
    )


UC_PARAMS = [pytest.param(case, id=_user_case_id(case)) for case in USER_CASES]


# 官方 74 例 user_case 矩阵，builtin causal 路径。
@pytest.mark.parametrize("case", UC_PARAMS)
def test_user_case_backward_precision(case, backends):
    _assert_case(case, BUILTIN, backends)


# 图模式：编译反向（autograd）vs PyTorch golden，no-mask 分支。
@pytest.mark.graph
@pytest.mark.parametrize("case", GRAPH_PARAMS)
def test_graph_backward_precision(case, backends):
    _, native, validator = backends
    data = _make_case_data(case)
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    _, gradients = run_compiled(case, data)
    rab_grad = gradients[3].cpu() if rab is not None else None
    actual = (gradients[0].cpu(), gradients[1].cpu(), gradients[2].cpu(), rab_grad)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.backward(grad, q, k, v, rab, None)
    passed, detail = validator.backward_verify(actual, expected)
    assert passed, detail


# 图模式：builtin causal 分支；context-target 变体额外把非空可选张量喂进被追踪的图。
@pytest.mark.graph
@pytest.mark.parametrize("case,with_context_target", GRAPH_CAUSAL_VARIANT_PARAMS)
def test_graph_causal_backward_precision(case, with_context_target, backends):
    _, native, validator = backends
    if with_context_target:
        context_value = min(GRAPH_CONTEXT_VALUE, case.max_seq_len // 2)
        case = replace(
            case,
            num_context=context_value,
            num_target=context_value,
            target_group_size=GRAPH_TARGET_GROUP_SIZE,
        )
    data = _make_case_data(case)
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    context, target = _hstu_parameters(case, f"npu:{DEVICE_ID}")
    _, gradients = run_compiled(case, data, MASK_TYPE_CAUSAL, 0, context, target)
    rab_grad = gradients[3].cpu() if rab is not None else None
    actual = (gradients[0].cpu(), gradients[1].cpu(), gradients[2].cpu(), rab_grad)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.backward(grad, q, k, v, rab, dense_mask)
    passed, detail = validator.backward_verify(actual, expected)
    assert passed, detail
