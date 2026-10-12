"""Autograd bridge from the forward attention plan to the backward plan."""

from typing import NamedTuple

import torch


class FwdAttnPlan(NamedTuple):
    tile: list[int]
    sparse_info: list[torch.Tensor]
    metadata: torch.Tensor | None
    use_aicpu: bool
    af: torch.Tensor | None


class BwdAttnPlan(NamedTuple):
    tile: list[int]
    sparse_info: list[torch.Tensor]
    metadata: torch.Tensor | None
    use_aicpu: bool


def get_fwd_attn_plan(
    q,
    k,
    v,
    af,
    seq_offset_q,
    seq_offset_k,
    max_seqlen_q,
    max_seqlen_k,
    num_context=None,
    num_target=None,
    target_group_size=1,
    window_size_left=-1,
    window_size_right=-1,
    mask_type=2,
    deterministic=True,
):
    tile, sparse, metadata, use_aicpu = torch.ops.ops_rec.get_fwd_attn_plan(
        q,
        k,
        v,
        af,
        seq_offset_q,
        seq_offset_k,
        max_seqlen_q,
        max_seqlen_k,
        num_context,
        num_target,
        target_group_size,
        window_size_left,
        window_size_right,
        mask_type,
        deterministic,
    )
    return FwdAttnPlan(tile, sparse, metadata, use_aicpu, af)


def get_bwd_attn_plan(
    q,
    k,
    v,
    af,
    seq_offset_q,
    seq_offset_k,
    max_seqlen_q,
    max_seqlen_k,
    num_context=None,
    num_target=None,
    target_group_size=1,
    window_size_left=-1,
    window_size_right=-1,
    mask_type=2,
    deterministic=True,
):
    tile, sparse, metadata, use_aicpu = torch.ops.ops_rec.get_bwd_attn_plan(
        q,
        k,
        v,
        af,
        seq_offset_q,
        seq_offset_k,
        max_seqlen_q,
        max_seqlen_k,
        num_context,
        num_target,
        target_group_size,
        window_size_left,
        window_size_right,
        mask_type,
        deterministic,
    )
    return BwdAttnPlan(tile, sparse, metadata, use_aicpu)


def hstu_v2(
    q,
    k,
    v,
    af,
    mask_type,
    max_seqlen_q,
    max_seqlen_k,
    silu_scale,
    seq_offset_q,
    seq_offset_k,
    rab=None,
    num_context=None,
    num_target=None,
    target_group_size=1,
    alpha=1.0,
    window_size_left=-1,
    window_size_right=-1,
    deterministic=True,
):
    """Execute ``get_fwd_attn_plan -> hstu_forward_v2`` with autograd."""
    plan = get_fwd_attn_plan(
        q,
        k,
        v,
        af,
        seq_offset_q,
        seq_offset_k,
        max_seqlen_q,
        max_seqlen_k,
        num_context,
        num_target,
        target_group_size,
        window_size_left,
        window_size_right,
        mask_type,
        deterministic,
    )
    return torch.ops.ops_rec.hstu_forward_v2(
        q,
        k,
        v,
        None,
        rab,
        mask_type,
        max_seqlen_q,
        max_seqlen_k,
        silu_scale,
        seq_offset_q,
        seq_offset_k,
        num_context,
        num_target,
        target_group_size,
        alpha,
        plan.metadata,
        plan.sparse_info or None,
        window_size_left,
        window_size_right,
        plan.af,
        deterministic,
    )


def _setup_context(ctx, inputs, output=None):
    values = list(inputs)
    values.extend([None] * (21 - len(values)))
    (
        q,
        k,
        v,
        _mask,
        rab,
        mask_type,
        max_q,
        max_k,
        scale,
        offsets_q,
        offsets_k,
        context,
        target,
        group_size,
        alpha,
        _metadata,
        sparse,
        win_left,
        win_right,
        af,
        deterministic,
    ) = values[:21]
    ctx.save_for_backward(q, k, v, rab, offsets_q, offsets_k, context, target, af)
    sparse_grad = [None] * len(sparse) if sparse is not None else None
    ctx.attrs = (mask_type, max_q, max_k, scale, group_size, alpha, win_left, win_right, deterministic, sparse_grad)


def _backward(ctx, grad):
    q, k, v, rab, offsets_q, offsets_k, context, target, af = ctx.saved_tensors
    mask_type, max_q, max_k, scale, group_size, alpha, win_left, win_right, deterministic, sparse_grad = ctx.attrs
    plan = get_bwd_attn_plan(
        q,
        k,
        v,
        af,
        offsets_q,
        offsets_k,
        max_q,
        max_k,
        context,
        target,
        group_size,
        win_left,
        win_right,
        mask_type,
        deterministic,
    )
    dq, dk, dv, drab = torch.ops.ops_rec.hstu_backward_v2(
        grad,
        q,
        k,
        v,
        max_q,
        max_k,
        offsets_q,
        offsets_k,
        rab,
        context,
        target,
        scale,
        group_size,
        alpha,
        win_left,
        win_right,
        plan.metadata,
        plan.sparse_info or None,
        deterministic,
    )
    gradients = (
        dq,
        dk,
        dv,
        None,
        drab if rab is not None else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        sparse_grad,
        None,
        None,
        None,
        None,
    )
    return gradients[: len(ctx.needs_input_grad)]


_registered = False


def register_autograd():
    global _registered
    if _registered:
        return
    torch.library.register_autograd(torch.ops.ops_rec.hstu_forward_v2.default, _backward, setup_context=_setup_context)
    _registered = True
