#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""hstu_v1 训练自动微分注册。

背景:
    hstu_v1 前向算子（hstu_dense / hstu_jagged）在 torch_extension 中不注册
    C++ AutogradPrivateUse1--C++ autograd Function 黑盒会导致 inductor 图模式
    数值错误，图模式仅走 PrivateUse1 + Meta 干净路径（与 paged/ln_mul/
    disentangle_attention 已验证模式一致）。反向梯度由独立算子
    hstu_dense_backward / hstu_jagged_backward 提供。

    本模块通过 torch.library.register_autograd 将 backward 算子绑定为前向算子
    的 autograd 公式（注册于 Autograd dispatch key，eager 与 torch.compile 均
    可用）。import ops_rec 后自动生效，训练时直接调用前向算子即可：

        y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
        loss = criterion(y, target)
        loss.backward()  # 梯度流向 q/k/v/attn_bias
"""

import logging

import torch

import ops_rec


# ---------------------------------------------------------------------------
# hstu_dense
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale)
# ---------------------------------------------------------------------------


def _dense_setup_context(ctx, inputs):
    q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale = inputs
    ctx.save_for_backward(q, k, v, mask, attn_bias)
    ctx.mask_type = mask_type
    ctx.max_seq_len = max_seq_len
    ctx.silu_scale = silu_scale


def _dense_backward(ctx, grad):
    q, k, v, mask, attn_bias = ctx.saved_tensors
    q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_dense_backward(
        grad, q, k, v, mask, attn_bias, ctx.mask_type, ctx.max_seq_len, ctx.silu_scale
    )
    # 与 hstu_dense schema 输入一一对应（8 个）；mask 无梯度；
    # 无 attn_bias 时算子返回的是 256 对齐的占位梯度，丢弃。
    return (
        q_grad,
        k_grad,
        v_grad,
        None,
        attn_bias_grad if attn_bias is not None else None,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# hstu_jagged（equal 语义：q/k 等长，seq_offset 共用）
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale,
#          seq_offset, num_context, num_target, target_group_size, alpha, deterministic)
# ---------------------------------------------------------------------------


def _jagged_equal_setup_context(ctx, inputs):
    (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale, seq_offset, num_context, num_target,
     target_group_size, alpha, deterministic) = inputs
    ctx.save_for_backward(q, k, v, mask, attn_bias, seq_offset, num_context, num_target)
    ctx.mask_type = mask_type
    ctx.max_seq_len = max_seq_len
    ctx.silu_scale = silu_scale
    ctx.target_group_size = target_group_size
    ctx.alpha = alpha


def _jagged_equal_backward(ctx, grad):
    q, k, v, mask, attn_bias, seq_offset, num_context, num_target = ctx.saved_tensors
    q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
        grad,
        q,
        k,
        v,
        mask,
        attn_bias,
        ctx.mask_type,
        ctx.max_seq_len,
        ctx.max_seq_len,
        ctx.silu_scale,
        seq_offset,
        seq_offset,
        num_context,
        num_target,
        ctx.target_group_size,
        ctx.alpha,
    )
    # 与 hstu_jagged.equal schema 输入一一对应（14 个）；无 attn_bias 时算子
    # 第 4 个返回值为 undefined tensor，丢弃。
    return (
        q_grad,
        k_grad,
        v_grad,
        None,
        attn_bias_grad if attn_bias is not None else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# hstu_jagged（delta 语义：q/k 可不等长，max_seq_len_k / seq_offset_k 独立）
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, max_seq_len_k,
#          silu_scale, seq_offset, seq_offset_k, num_context, num_target,
#          target_group_size, alpha, deterministic)
# ---------------------------------------------------------------------------


def _jagged_delta_setup_context(ctx, inputs):
    (q, k, v, mask, attn_bias, mask_type, max_seq_len, max_seq_len_k, silu_scale, seq_offset, seq_offset_k,
     num_context, num_target, target_group_size, alpha, deterministic) = inputs
    ctx.save_for_backward(q, k, v, mask, attn_bias, seq_offset, seq_offset_k, num_context, num_target)
    ctx.mask_type = mask_type
    ctx.max_seq_len = max_seq_len
    ctx.max_seq_len_k = max_seq_len_k
    ctx.silu_scale = silu_scale
    ctx.target_group_size = target_group_size
    ctx.alpha = alpha


def _jagged_delta_backward(ctx, grad):
    q, k, v, mask, attn_bias, seq_offset, seq_offset_k, num_context, num_target = ctx.saved_tensors
    q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
        grad,
        q,
        k,
        v,
        mask,
        attn_bias,
        ctx.mask_type,
        ctx.max_seq_len,
        ctx.max_seq_len_k,
        ctx.silu_scale,
        seq_offset,
        seq_offset_k,
        num_context,
        num_target,
        ctx.target_group_size,
        ctx.alpha,
    )
    # 与 hstu_jagged.delta schema 输入一一对应（16 个）。
    return (
        q_grad,
        k_grad,
        v_grad,
        None,
        attn_bias_grad if attn_bias is not None else None,
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
    )


_registered = False


def register_autograd():
    """hstu_v1 autograd 注册入口（由 ops_rec.autograd 包加载时自动调用）。

    前向算子保持干净注册路径（PrivateUse1 + Meta），自动微分公式通过
    torch.library.register_autograd 注册在 Autograd dispatch key 上，
    eager 与 torch.compile 训练均可直接使用前向算子 + loss.backward()。
    注册重复（如进程内二次调用）仅告警不抛错，与 fbgemm-ascend
    register_python_autograd() 的容错行为一致。
    """
    global _registered
    if _registered:
        return
    try:
        torch.library.register_autograd(
            "ops_rec::hstu_dense",
            _dense_backward,
            setup_context=_dense_setup_context,
        )
        torch.library.register_autograd(
            "ops_rec::hstu_jagged",
            _jagged_equal_backward,
            setup_context=_jagged_equal_setup_context,
            overload_name="equal",
        )
        torch.library.register_autograd(
            "ops_rec::hstu_jagged",
            _jagged_delta_backward,
            setup_context=_jagged_delta_setup_context,
            overload_name="delta",
        )
    except RuntimeError as e:
        logging.warning("ops_rec.autograd.hstu_v1: autograd registration skipped: %s", e)
        return
    _registered = True
