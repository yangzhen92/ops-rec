"""Guide12 deterministic/Stream-K precision and autograd matrices (252 cases)."""

import itertools
import os
import pytest
import torch
import torch_npu  # noqa: F401
import ops_rec  # noqa: F401
from ops_rec.ascendc.attention.hstu_v2 import hstu_v2

DEVICE = f"npu:{int(os.getenv('ASCEND_DEVICE_ID', '0'))}"
DTYPES = (torch.float16, torch.bfloat16)
DIMS = (32, 64, 128, 256)
MASKS = ("none", "causal", "context_target", "local", "arbitrary")


def _inputs(q_len, k_len, heads, dq, dv, dtype, seed, rab=False, rab_heads=None):
    torch.manual_seed(seed)
    q = torch.empty(q_len, heads, dq, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)
    k = torch.empty(k_len, heads, dq, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)
    v = torch.empty(k_len, heads, dv, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)
    g = torch.empty(q_len, heads, dv, dtype=dtype, device=DEVICE).uniform_(-0.1, 0.1)
    oq = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
    ok = torch.tensor([0, k_len], dtype=torch.int32, device=DEVICE)
    bias = None
    if rab:
        bias = torch.empty(1, rab_heads or heads, q_len, k_len, dtype=dtype, device=DEVICE).uniform_(-0.05, 0.05)
    return q, k, v, g, oq, ok, bias


def _mask_args(kind, ok, k_len):
    if kind == "none":
        return None, 2, None, None, -1, -1
    if kind == "causal":
        return None, 0, None, None, -1, 0
    if kind == "context_target":
        return (
            None,
            0,
            torch.tensor([16], dtype=torch.int32, device=DEVICE),
            torch.tensor([32], dtype=torch.int32, device=DEVICE),
            -1,
            0,
        )
    if kind == "local":
        return None, 0, None, None, 64, 0
    zeros = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    af = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(ok, zeros, zeros, 1, k_len)
    return af, 0, None, None, -1, 0


def _forward(case, deterministic):
    q, k, v, _, oq, ok, rab = case[:7]
    af, mask_type, context, target, left, right = _mask_args(case.mask, ok, k.size(0))
    return hstu_v2(
        q,
        k,
        v,
        af,
        mask_type,
        q.size(0),
        k.size(0),
        1 / q.size(0),
        oq,
        ok,
        rab=rab,
        num_context=context,
        num_target=target,
        target_group_size=4,
        alpha=0.5,
        window_size_left=left,
        window_size_right=right,
        deterministic=deterministic,
    )


def _backward(case, deterministic):
    q, k, v, grad, oq, ok, rab = case[:7]
    af, _, context, target, left, right = _mask_args(case.mask, ok, k.size(0))
    plan = torch.ops.ops_rec.get_bwd_attn_plan(
        q,
        k,
        v,
        af,
        oq,
        ok,
        q.size(0),
        k.size(0),
        context,
        target,
        4,
        left,
        right,
        0 if case.mask != "none" else 2,
        deterministic,
    )
    return torch.ops.ops_rec.hstu_backward_v2(
        grad,
        q,
        k,
        v,
        q.size(0),
        k.size(0),
        oq,
        ok,
        rab,
        context,
        target,
        1 / q.size(0),
        4,
        0.5,
        left,
        right,
        plan[2],
        plan[1] or None,
        deterministic,
    )


class Case(tuple):
    __slots__ = ()
    mask = property(lambda self: self[7])


def _case(q, k, heads, dq, dv, dtype, rab, mask, seed, rab_heads=None):
    return Case((*_inputs(q, k, heads, dq, dv, dtype, seed, rab, rab_heads), mask))


CORE = list(itertools.product(("forward", "backward"), MASKS, (False, True), DTYPES, DIMS))
assert len(CORE) == 160


@pytest.mark.parametrize("direction,mask,rab,dtype,dim", CORE)
def test_stream_k_core_matrix(direction, mask, rab, dtype, dim):
    case = _case(128, 3584, 1, dim, dim, dtype, rab, mask, 202612000 + dim + 13 * len(mask))
    reference = _forward(case, True) if direction == "forward" else _backward(case, True)
    actual = _forward(case, False) if direction == "forward" else _backward(case, False)
    torch.npu.synchronize()
    reference = (reference,) if direction == "forward" else reference
    actual = (actual,) if direction == "forward" else actual
    for got, want in zip(actual, reference):
        if got is not None and want is not None:
            assert torch.allclose(got.cpu(), want.cpu(), rtol=1e-3, atol=1e-3)


SUPPLEMENT = []
for direction, dtype, ratio in itertools.product(("forward", "backward"), DTYPES, (4, 8, 16, 28)):
    SUPPLEMENT.append((direction, "none", 33, 33 * ratio, 64, 64, dtype, False, None))
for direction, dtype, dims in itertools.product(("forward", "backward"), DTYPES, ((128, 64), (256, 128))):
    SUPPLEMENT.append((direction, "none", 128, 3584, *dims, dtype, False, None))
for direction, dtype, mask in itertools.product(("forward", "backward"), DTYPES, ("none", "arbitrary")):
    SUPPLEMENT.append((direction, mask, 128, 3584, 64, 64, dtype, True, 1))
for direction, dtype, q_len in itertools.product(("forward", "backward"), DTYPES, (33, 65)):
    SUPPLEMENT.append((direction, "none", q_len, q_len * 28, 64, 64, dtype, False, None))
assert len(SUPPLEMENT) == 40


@pytest.mark.parametrize("direction,mask,q_len,k_len,dq,dv,dtype,rab,rab_heads", SUPPLEMENT)
def test_stream_k_supplement(direction, mask, q_len, k_len, dq, dv, dtype, rab, rab_heads):
    case = _case(q_len, k_len, 2 if rab_heads else 1, dq, dv, dtype, rab, mask, 202612500 + q_len + dq, rab_heads)
    reference = _forward(case, True) if direction == "forward" else _backward(case, True)
    actual = _forward(case, False) if direction == "forward" else _backward(case, False)
    torch.npu.synchronize()
    reference = (reference,) if direction == "forward" else reference
    actual = (actual,) if direction == "forward" else actual
    for got, want in zip(actual, reference):
        if got is not None and want is not None:
            assert torch.allclose(got.cpu(), want.cpu(), rtol=1e-3, atol=1e-3)


AUTOGRAD = []
for deterministic, mask, dtype, dim in itertools.product(
    (True, False), ("none", "causal", "arbitrary"), DTYPES, (32, 128, 256)
):
    AUTOGRAD.append((deterministic, mask, 1024, 1024, dim, dim, dtype, False, None))
for deterministic, mask, dtype in itertools.product((True, False), ("none", "arbitrary"), DTYPES):
    AUTOGRAD.append((deterministic, mask, 128, 3584, 64, 64, dtype, False, None))
for dtype, dims in itertools.product(DTYPES, ((128, 64), (256, 128))):
    AUTOGRAD.append((False, "none", 1024, 1024, *dims, dtype, False, None))
for dtype, mask in itertools.product(DTYPES, ("none", "arbitrary")):
    AUTOGRAD.append((False, mask, 1024, 1024, 64, 64, dtype, True, 1))
assert len(AUTOGRAD) == 52


@pytest.mark.parametrize("deterministic,mask,q_len,k_len,dq,dv,dtype,rab,rab_heads", AUTOGRAD)
def test_stream_k_autograd_matches_direct(deterministic, mask, q_len, k_len, dq, dv, dtype, rab, rab_heads):
    case = _case(q_len, k_len, 1, dq, dv, dtype, rab, mask, 202613000 + q_len + dq, rab_heads)
    q, k, v, grad, *_ = case[:7]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    output = _forward(case, deterministic)
    output.backward(grad)
    direct = _backward(case, deterministic)
    torch.npu.synchronize()
    for got, want in zip((q.grad, k.grad, v.grad), direct[:3]):
        assert torch.allclose(got.cpu(), want.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "direction,q_len,k_len,mask",
    (
        ("forward", 128, 3584, "none"),
        ("forward", 128, 3584, "arbitrary"),
        ("backward", 3584, 128, "none"),
        ("backward", 1024, 1024, "causal"),
        ("backward", 1024, 1024, "arbitrary"),
    ),
)
def test_stream_k_uses_aicpu_metadata(direction, q_len, k_len, mask):
    """The nondeterministic long-inner-axis path must consume Stream-K metadata."""
    case = _case(q_len, k_len, 1, 64, 64, torch.float16, False, mask, 202614007)
    q, k, v, grad, oq, ok, _ = case[:7]
    af, mask_type, context, target, left, right = _mask_args(mask, ok, k_len)
    plan_op = torch.ops.ops_rec.get_fwd_attn_plan if direction == "forward" else torch.ops.ops_rec.get_bwd_attn_plan
    _, sparse, metadata, use_aicpu = plan_op(
        q,
        k,
        v,
        af,
        oq,
        ok,
        q_len,
        k_len,
        context,
        target,
        4,
        left,
        right,
        mask_type,
        False,
    )
    assert use_aicpu and metadata is not None
    assert int(metadata.cpu()[15]) == 0x534B4855
    reference = _forward(case, True) if direction == "forward" else _backward(case, True)
    if direction == "forward":
        actual = torch.ops.ops_rec.hstu_forward_v2(
            q,
            k,
            v,
            None,
            None,
            mask_type,
            q_len,
            k_len,
            1 / q_len,
            oq,
            ok,
            context,
            target,
            4,
            0.5,
            metadata,
            sparse or None,
            left,
            right,
            af,
            False,
        )
        actual, reference = (actual,), (reference,)
    else:
        actual = torch.ops.ops_rec.hstu_backward_v2(
            grad,
            q,
            k,
            v,
            q_len,
            k_len,
            oq,
            ok,
            None,
            context,
            target,
            1 / q_len,
            4,
            0.5,
            left,
            right,
            metadata,
            sparse or None,
            False,
        )
    torch.npu.synchronize()
    for got, want in zip(actual, reference):
        if got is not None and want is not None:
            assert torch.allclose(got.cpu(), want.cpu(), rtol=1e-3, atol=1e-3)
