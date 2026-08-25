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

兼容性说明（跨 torch 版本差异，均已在代码中适配）:
    - 注册入参使用显式 OpOverload（hstu_dense.default / hstu_jagged.equal /
      hstu_jagged.delta）。不可传 OpOverloadPacket（torch 2.1/2.2 的
      register_autograd 仅接受 Union[OpOverload, str]），也不可用字符串 +
      overload_name 参数（该参数仅部分版本存在，2.5 起已移除）。
    - setup_context 签名为 (ctx, inputs, output) 三参数形式（output 带默认值，
      兼容仅传两参的调用方）。
    - inputs 元组长度与 backward 返回梯度数量按 ctx.needs_input_grad 自适应
      （不同版本传入完整 schema 参数或调用时参数，长度可能不同）。
"""

import logging

import torch

import ops_rec

logger = logging.getLogger(__name__)

# 各算子 schema 的输入数量（用于防御性解包）
_DENSE_NARGS = 8
_JAGGED_EQUAL_NARGS = 14
_JAGGED_DELTA_NARGS = 16


def _unpack(inputs, n):
    """按 schema 前 n 个输入防御性解包（inputs 长度不足时以 None 补齐）。"""
    vals = list(inputs)
    if len(vals) < n:
        vals.extend([None] * (n - len(vals)))
    return tuple(vals[:n])


def _fit_grads(grads, ctx):
    """按 forward 实际参数数量截断梯度元组（与 fbgemm-ascend 同款技巧）。

    autograd 要求 backward 返回梯度数量与 forward 参数数量一致；
    ctx.needs_input_grad 长度即 forward 参数数量（不同 torch 版本可能传入
    完整 schema 参数或仅调用时参数）。
    """
    n = len(ctx.needs_input_grad)
    grads = tuple(grads)
    if len(grads) < n:
        grads = grads + (None,) * (n - len(grads))
    return grads[:n]


# ---------------------------------------------------------------------------
# hstu_dense
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale)
# ---------------------------------------------------------------------------


def _dense_setup_context(ctx, inputs, output=None):
    q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale = _unpack(inputs, _DENSE_NARGS)
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
    return _fit_grads(
        (
            q_grad,
            k_grad,
            v_grad,
            None,
            attn_bias_grad if attn_bias is not None else None,
            None,
            None,
            None,
        ),
        ctx,
    )


# ---------------------------------------------------------------------------
# hstu_jagged（equal 语义：q/k 等长，seq_offset 共用）
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale,
#          seq_offset, num_context, num_target, target_group_size, alpha, deterministic)
# ---------------------------------------------------------------------------


def _jagged_equal_setup_context(ctx, inputs, output=None):
    (q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale, seq_offset, num_context, num_target,
     target_group_size, alpha, deterministic) = _unpack(inputs, _JAGGED_EQUAL_NARGS)
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
    return _fit_grads(
        (
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
        ),
        ctx,
    )


# ---------------------------------------------------------------------------
# hstu_jagged（delta 语义：q/k 可不等长，max_seq_len_k / seq_offset_k 独立）
# schema: (q, k, v, mask, attn_bias, mask_type, max_seq_len, max_seq_len_k,
#          silu_scale, seq_offset, seq_offset_k, num_context, num_target,
#          target_group_size, alpha, deterministic)
# ---------------------------------------------------------------------------


def _jagged_delta_setup_context(ctx, inputs, output=None):
    (q, k, v, mask, attn_bias, mask_type, max_seq_len, max_seq_len_k, silu_scale, seq_offset, seq_offset_k,
     num_context, num_target, target_group_size, alpha, deterministic) = _unpack(inputs, _JAGGED_DELTA_NARGS)
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
    return _fit_grads(
        (
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
        ),
        ctx,
    )


# 注册状态（供诊断）：成功注册的 op 名与失败原因
_registered = False
_registration_errors = {}


def _register_one(name, op_overload, backward, setup_context):
    """注册单个算子的 autograd 公式，失败记录原因并返回 False（不影响其余算子）。"""
    try:
        torch.library.register_autograd(op_overload, backward, setup_context=setup_context)
    except Exception as e:  # noqa: BLE001 - 跨版本 API 差异需兜底记录
        _registration_errors[name] = f"{type(e).__name__}: {e}"
        logger.warning("ops_rec.autograd.hstu_v1: register '%s' failed: %s: %s", name, type(e).__name__, e)
        return False
    logger.info("ops_rec.autograd.hstu_v1: registered autograd for '%s'", name)
    return True


def register_autograd():
    """hstu_v1 autograd 注册入口（由 ops_rec.autograd 包加载时自动调用）。

    前向算子保持干净注册路径（PrivateUse1 + Meta），自动微分公式通过
    torch.library.register_autograd 注册在 Autograd dispatch key 上，
    eager 与 torch.compile 训练均可直接使用前向算子 + loss.backward()。

    注册入参为显式 OpOverload（.default/.equal/.delta），逐算子独立注册、
    互不影响；失败仅告警并记录到 _registration_errors（可通过
    ops_rec.autograd.hstu_v1_autograd._registration_errors 查询诊断），
    与 fbgemm-ascend register_python_autograd() 的容错行为一致。
    """
    global _registered
    if _registered:
        return
    ok = True
    ok &= _register_one(
        "hstu_dense",
        torch.ops.ops_rec.hstu_dense.default,
        _dense_backward,
        setup_context=_dense_setup_context,
    )
    ok &= _register_one(
        "hstu_jagged.equal",
        torch.ops.ops_rec.hstu_jagged.equal,
        _jagged_equal_backward,
        setup_context=_jagged_equal_setup_context,
    )
    ok &= _register_one(
        "hstu_jagged.delta",
        torch.ops.ops_rec.hstu_jagged.delta,
        _jagged_delta_backward,
        setup_context=_jagged_delta_setup_context,
    )
    if ok:
        _registered = True
