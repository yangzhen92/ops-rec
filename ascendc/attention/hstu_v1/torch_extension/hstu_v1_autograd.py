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
"""hstu_v1 训练自动微分包装。

背景:
    hstu_v1 前向算子（hstu_dense / hstu_jagged.equal|delta）在 torch_extension 中
    不注册 C++ AutogradPrivateUse1--autograd Function 黑盒会导致 inductor 图模式
    数值错误，图模式仅走 PrivateUse1 + Meta 干净路径（与 paged/ln_mul/
    disentangle_attention 已验证模式一致）。反向梯度由独立算子
    hstu_dense_backward / hstu_jagged_backward.equal|delta 提供。

    如需训练自动微分（loss.backward()），请使用本模块提供的
    torch.autograd.Function 子类包装 forward + backward 算子。

    import ops_rec 后自动加载本模块，通过
    ops_rec.autograd.hstu_v1_autograd.HstuDenseAutograd 等访问。

用法:
    from ops_rec.autograd.hstu_v1_autograd import HstuDenseAutograd

    q.requires_grad_(True)
    y = HstuDenseAutograd.apply(q, k, v, None, None, 0, S, 1.0 / S)
    loss = criterion(y, target)
    loss.backward()  # 梯度流向 q/k/v/bias
"""

import torch

import ops_rec


class HstuDenseAutograd(torch.autograd.Function):
    """dense 布局 hstu_dense 的自动微分包装。

    forward 调用 ops_rec.ascendc.attention.hstu_dense（PrivateUse1 + Meta 注册），
    backward 调用 ops_rec.ascendc.attention.hstu_dense_backward。
    """

    @staticmethod
    def forward(ctx, q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale):
        ctx.save_for_backward(q, k, v, mask, attn_bias)
        ctx.mask_type = mask_type
        ctx.max_seq_len = max_seq_len
        ctx.silu_scale = silu_scale
        return ops_rec.ascendc.attention.hstu_dense(
            q,
            k,
            v,
            mask=mask,
            attn_bias=attn_bias,
            mask_type=mask_type,
            max_seq_len=max_seq_len,
            silu_scale=silu_scale,
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, mask, attn_bias = ctx.saved_tensors
        q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_dense_backward(
            grad_output,
            q,
            k,
            v,
            mask,
            attn_bias,
            ctx.mask_type,
            ctx.max_seq_len,
            ctx.silu_scale,
        )
        # 返回值顺序与 forward 入参一一对应: q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale
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


class HstuJaggedAutograd(torch.autograd.Function):
    """jagged 布局 hstu_jagged (equal 语义) 的自动微分包装。

    forward 调用 ops_rec.ascendc.attention.hstu_jagged，
    backward 调用 ops_rec.ascendc.attention.hstu_jagged_backward
    （equal 语义: max_seq_len_k = max_seq_len, seq_offset_k = seq_offset）。
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        attn_bias,
        mask_type,
        max_seq_len,
        silu_scale,
        seq_offset,
        num_context,
        num_target,
        target_group_size,
        alpha,
    ):
        ctx.save_for_backward(q, k, v, mask, attn_bias, seq_offset, num_context, num_target)
        ctx.mask_type = mask_type
        ctx.max_seq_len = max_seq_len
        ctx.silu_scale = silu_scale
        ctx.target_group_size = target_group_size
        ctx.alpha = alpha
        return ops_rec.ascendc.attention.hstu_jagged(
            q,
            k,
            v,
            mask,
            attn_bias,
            mask_type,
            max_seq_len,
            max_seq_len,
            silu_scale,
            seq_offset,
            seq_offset,
            num_context,
            num_target,
            target_group_size,
            alpha,
            False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, mask, attn_bias, seq_offset, num_context, num_target = ctx.saved_tensors
        q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
            grad_output,
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
        # 返回值顺序与 forward 入参一一对应（13 个入参）
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
        )


class HstuJaggedDeltaAutograd(torch.autograd.Function):
    """jagged 布局 hstu_jagged (delta 语义) 的自动微分包装。

    forward 调用 ops_rec.ascendc.attention.hstu_jagged，
    backward 调用 ops_rec.ascendc.attention.hstu_jagged_backward
    （delta 语义: q/k 可不等长，max_seq_len_k 与 seq_offset_k 独立）。
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        attn_bias,
        mask_type,
        max_seq_len,
        max_seq_len_k,
        silu_scale,
        seq_offset,
        seq_offset_k,
        num_context,
        num_target,
        target_group_size,
        alpha,
        deterministic,
    ):
        ctx.save_for_backward(q, k, v, mask, attn_bias, seq_offset, seq_offset_k, num_context, num_target)
        ctx.mask_type = mask_type
        ctx.max_seq_len = max_seq_len
        ctx.max_seq_len_k = max_seq_len_k
        ctx.silu_scale = silu_scale
        ctx.target_group_size = target_group_size
        ctx.alpha = alpha
        return ops_rec.ascendc.attention.hstu_jagged(
            q,
            k,
            v,
            mask,
            attn_bias,
            mask_type,
            max_seq_len,
            max_seq_len_k,
            silu_scale,
            seq_offset,
            seq_offset_k,
            num_context,
            num_target,
            target_group_size,
            alpha,
            deterministic,
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, mask, attn_bias, seq_offset, seq_offset_k, num_context, num_target = ctx.saved_tensors
        q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
            grad_output,
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
        # 返回值顺序与 forward 入参一一对应（16 个入参）
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


def register_autograd():
    """hstu_v1 autograd 包装注册入口（由 ops_rec.autograd 自动调用）。

    前向算子保持干净路径（不注册 C++ AutogradPrivateUse1，保证 inductor
    图模式仅走 PrivateUse1 + Meta）。自动微分由本模块的
    torch.autograd.Function 子类提供，import ops_rec 后即可通过
    ops_rec.autograd.hstu_v1_autograd.HstuDenseAutograd 等访问。
    """
