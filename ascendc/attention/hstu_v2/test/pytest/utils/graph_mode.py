"""Shared torch.compile (graph mode) machinery for the HSTU V2 precision suites.

Graph-mode risk is compile/autograd-bridge related, not numeric, so the case
selection reuses the precision feature matrix and only spans the branches that
change the traced graph: optional rab, both dtypes, multi-batch, multi-head,
k != q, and dim_v != dim_qk.
"""

import os
import sys

import pytest
import torch
import torch._dynamo

import ops_rec  # noqa: F401 - loads the op package and registers autograd bridges
from ops_rec.ascendc.attention.hstu_v2 import hstu_v2
from utils.feature_matrix import case_id, precision_cases, pytest_params

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


def graph_cases():
    """Pick small cases that maximize feature-combo coverage for graph mode."""
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


def _first_per_combo(cases):
    """First case of each (has_rab, data_type) combo, in matrix order."""
    seen, selected = set(), []
    for case in cases:
        key = (case.has_rab, case.data_type)
        if key not in seen:
            seen.add(key)
            selected.append(case)
    return selected


GRAPH_CASES = graph_cases()
GRAPH_PARAMS = pytest_params(GRAPH_CASES, "graph-")

# Causal variants: one case per rab×dtype combo, run with mask_type=0 so the
# traced graph covers the builtin-mask branch that beta's graph suite cannot
# express.  Context/target variants additionally feed non-empty optional
# tensors into the graph; batch-1 bases keep the 64/64 clamp below effective.
GRAPH_CAUSAL_CASES = _first_per_combo(GRAPH_CASES)
GRAPH_CONTEXT_BASES = [case for case in GRAPH_CASES if case.batch_size == 1 and case.max_seq_len >= 128]
GRAPH_CONTEXT_CASES = _first_per_combo(GRAPH_CONTEXT_BASES)[:2]


def _graph_causal_variant_params():
    """(case, with_context_target) pairs covering both causal graph variants."""
    variants = [(case, False) for case in GRAPH_CAUSAL_CASES]
    variants += [(case, True) for case in GRAPH_CONTEXT_CASES]
    selected = []
    for case, with_context_target in variants:
        prefix = "graph-context-" if with_context_target else "graph-causal-"
        selected.append(pytest.param(case, with_context_target, id=case_id(case, prefix)))
    return selected


GRAPH_CAUSAL_VARIANT_PARAMS = _graph_causal_variant_params()


def torch_compiled(fn, **kwargs):
    """torch.compile wrapper; Python >= 3.12 falls back to eager execution."""
    if sys.version_info < (3, 12, 0):
        return torch.compile(fn, **kwargs)
    return fn


def run_compiled(case, data, mask_type=MASK_TYPE_NO_MASK, window_right=-1, context=None, target=None):
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    compiled_fn = torch_compiled(hstu_v2, dynamic=True, fullgraph=True)
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
