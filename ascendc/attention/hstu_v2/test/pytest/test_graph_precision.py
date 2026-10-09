"""torch.compile (inductor, fullgraph) precision for the plan-based HSTU V2 entry point."""

import os
import sys
from dataclasses import replace

import pytest
import torch
import torch._dynamo

import ops_rec  # noqa: F401 - loads the op package and registers autograd bridges
from ops_rec.ascendc.attention.hstu_v2 import hstu_v2
from utils.feature_matrix import (
    _hstu_parameters,
    _make_case_data,
    create_backends,
    precision_cases,
    pytest_params,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
DEVICE = f"npu:{DEVICE_ID}"
ALPHA = 0.5
SCALE = 1 / 1024
MASK_TYPE_NO_MASK = 2
MASK_TYPE_CAUSAL = 0
GRAPH_CONTEXT_VALUE = 64
GRAPH_TARGET_GROUP_SIZE = 16

# Distinct shapes accumulate separate compiled graphs; raise the dynamo
# recompile limits so a full run does not hit FailOnRecompileLimitHit.
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 1024


def _graph_cases():
    """Pick small cases that maximize feature-combo coverage for graph mode.

    Graph-mode risk is compile/autograd-bridge related, not numeric, so the
    selection spans the branches that change the traced graph: optional rab,
    both dtypes, multi-batch, multi-head, k != q, and dim_v != dim_qk.
    """
    small = [case for case in precision_cases() if case.max_seq_len <= 512]

    def features(case):
        return (
            case.has_rab,
            case.data_type,
            case.batch_size > 1,
            case.head_num > 1,
            case.head_dim_v is not None,
            case.max_seq_len_k is not None,
        )

    selected, seen = [], set()
    for case in small:
        key = features(case)
        if key not in seen:
            seen.add(key)
            selected.append(case)
    return selected


CASES = _graph_cases()
PARAMS = pytest_params(CASES, "graph-")


def _first_per_combo(cases):
    """First case of each (has_rab, data_type) combo, in matrix order."""
    seen, selected = set(), []
    for case in cases:
        key = (case.has_rab, case.data_type)
        if key not in seen:
            seen.add(key)
            selected.append(case)
    return selected


# Causal variants: one case per rab×dtype combo, run with mask_type=0 so the
# traced graph covers the builtin-mask branch that beta's graph suite cannot
# express.  Context/target variants additionally feed non-empty optional
# tensors into the graph; batch-1 bases keep the 64/64 clamp below effective.
CAUSAL_CASES = _first_per_combo(CASES)
CONTEXT_BASES = [case for case in CASES if case.batch_size == 1 and case.max_seq_len >= 128]
CONTEXT_CASES = _first_per_combo(CONTEXT_BASES)[:2]
CAUSAL_PARAMS = pytest_params(CAUSAL_CASES, "graph-causal-")
CONTEXT_PARAMS = pytest_params(CONTEXT_CASES, "graph-context-")


def _torch_compiled(fn, **kwargs):
    """torch.compile wrapper; Python >= 3.12 falls back to eager execution."""
    if sys.version_info < (3, 12, 0):
        return torch.compile(fn, **kwargs)
    return fn


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


def _run_compiled(case, data, mask_type=MASK_TYPE_NO_MASK, window_right=-1, context=None, target=None):
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    compiled_fn = _torch_compiled(hstu_v2, dynamic=True, fullgraph=True)
    grad_npu = grad.to(DEVICE)
    q_npu = q.to(DEVICE).requires_grad_(True)
    k_npu = k.to(DEVICE).requires_grad_(True)
    v_npu = v.to(DEVICE).requires_grad_(True)
    rab_npu = rab.to(DEVICE).requires_grad_(True) if rab is not None else None
    offsets_q_npu = torch.as_tensor(offsets_q, dtype=torch.int32, device=DEVICE)
    offsets_k_npu = torch.as_tensor(offsets_k, dtype=torch.int32, device=DEVICE)
    output = compiled_fn(
        q_npu,
        k_npu,
        v_npu,
        None,
        mask_type,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        SCALE,
        offsets_q_npu,
        offsets_k_npu,
        rab_npu,
        context,
        target,
        case.target_group_size,
        ALPHA,
        -1,
        window_right,
        True,
    )
    torch.npu.synchronize()
    inputs = (q_npu, k_npu, v_npu, rab_npu) if rab_npu is not None else (q_npu, k_npu, v_npu)
    gradients = torch.autograd.grad(output, inputs, grad_npu)
    return output, gradients


@pytest.mark.graph
@pytest.mark.parametrize("case", PARAMS)
def test_graph_forward_precision(case, backends):
    _, native, validator = backends
    data = _make_case_data(case)
    _, q, k, v, rab, _, offsets_q, offsets_k = data
    output, _ = _run_compiled(case, data)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.forward(q, k, v, rab, None)
    passed, detail = validator.forward_verify(output.cpu(), expected)
    assert passed, detail


@pytest.mark.graph
@pytest.mark.parametrize("case", PARAMS)
def test_graph_backward_precision(case, backends):
    _, native, validator = backends
    data = _make_case_data(case)
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    _, gradients = _run_compiled(case, data)
    rab_grad = gradients[3].cpu() if rab is not None else None
    actual = (gradients[0].cpu(), gradients[1].cpu(), gradients[2].cpu(), rab_grad)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.backward(grad, q, k, v, rab, None)
    passed, detail = validator.backward_verify(actual, expected)
    assert passed, detail


@pytest.mark.graph
@pytest.mark.parametrize("case", CAUSAL_PARAMS)
def test_graph_causal_forward_precision(case, backends):
    _, native, validator = backends
    data = _make_case_data(case)
    _, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    context, target = _hstu_parameters(case, DEVICE)
    output, _ = _run_compiled(case, data, MASK_TYPE_CAUSAL, 0, context, target)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.forward(q, k, v, rab, dense_mask)
    passed, detail = validator.forward_verify(output.cpu(), expected)
    assert passed, detail


@pytest.mark.graph
@pytest.mark.parametrize("case", CAUSAL_PARAMS)
def test_graph_causal_backward_precision(case, backends):
    _, native, validator = backends
    data = _make_case_data(case)
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    context, target = _hstu_parameters(case, DEVICE)
    _, gradients = _run_compiled(case, data, MASK_TYPE_CAUSAL, 0, context, target)
    rab_grad = gradients[3].cpu() if rab is not None else None
    actual = (gradients[0].cpu(), gradients[1].cpu(), gradients[2].cpu(), rab_grad)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.backward(grad, q, k, v, rab, dense_mask)
    passed, detail = validator.backward_verify(actual, expected)
    assert passed, detail


@pytest.mark.graph
@pytest.mark.parametrize("case", CONTEXT_PARAMS)
def test_graph_context_target_forward_precision(case, backends):
    _, native, validator = backends
    context_value = min(GRAPH_CONTEXT_VALUE, case.max_seq_len // 2)
    case = replace(
        case,
        num_context=context_value,
        num_target=context_value,
        target_group_size=GRAPH_TARGET_GROUP_SIZE,
    )
    data = _make_case_data(case)
    _, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    context, target = _hstu_parameters(case, DEVICE)
    output, _ = _run_compiled(case, data, MASK_TYPE_CAUSAL, 0, context, target)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.forward(q, k, v, rab, dense_mask)
    passed, detail = validator.forward_verify(output.cpu(), expected)
    assert passed, detail


@pytest.mark.graph
@pytest.mark.parametrize("case", CONTEXT_PARAMS)
def test_graph_context_target_backward_precision(case, backends):
    _, native, validator = backends
    context_value = min(GRAPH_CONTEXT_VALUE, case.max_seq_len // 2)
    case = replace(
        case,
        num_context=context_value,
        num_target=context_value,
        target_group_size=GRAPH_TARGET_GROUP_SIZE,
    )
    data = _make_case_data(case)
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    context, target = _hstu_parameters(case, DEVICE)
    _, gradients = _run_compiled(case, data, MASK_TYPE_CAUSAL, 0, context, target)
    rab_grad = gradients[3].cpu() if rab is not None else None
    actual = (gradients[0].cpu(), gradients[1].cpu(), gradients[2].cpu(), rab_grad)
    native_kernel = native.kernel(
        ALPHA, SCALE, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    expected = native_kernel.backward(grad, q, k, v, rab, dense_mask)
    passed, detail = validator.backward_verify(actual, expected)
    assert passed, detail


def test_graph_matrix():
    dtype_combos = {(case.has_rab, case.data_type) for case in CASES}
    assert dtype_combos == {
        (False, torch.float16),
        (False, torch.bfloat16),
        (True, torch.float16),
        (True, torch.bfloat16),
    }
    assert {(case.has_rab, case.data_type) for case in CAUSAL_CASES} == dtype_combos
    assert len(CONTEXT_CASES) == 2
