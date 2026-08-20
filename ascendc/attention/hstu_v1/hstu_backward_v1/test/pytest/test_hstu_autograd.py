#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""
训练自动微分回归用例。

hstu_v1 前向算子不注册 C++ AutogradPrivateUse1（保证 inductor 图模式干净），
训练场景的自动微分由 Python 侧 torch.autograd.Function 包装 forward + backward
算子提供。本用例校验：

1. eager：y.backward() 得到的 q/k/v/bias 梯度与直接调用 hstu_*_backward 算子一致；
2. graph：Python 侧包装可被 dynamo 追踪（forward 走 Meta 构图），eager vs compiled 梯度一致。
"""

import sys

import pytest
import torch

import ops_rec

from test_common_utils import allclose, MaskType

# pylint: disable=duplicate-code
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


# ---------------------------------------------------------------------------
# Python 侧自动微分包装（与 examples/torch/test_torch_hstu_backward_v1_autograd.py 中一致）
# ---------------------------------------------------------------------------

class HstuDenseAutograd(torch.autograd.Function):
    """dense 布局 hstu_dense 的自动微分包装。"""

    @staticmethod
    def forward(ctx, q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale):
        ctx.save_for_backward(q, k, v, mask, attn_bias)
        ctx.mask_type = mask_type
        ctx.max_seq_len = max_seq_len
        ctx.silu_scale = silu_scale
        return ops_rec.ascendc.attention.hstu_dense(
            q, k, v, mask=mask, attn_bias=attn_bias,
            mask_type=mask_type, max_seq_len=max_seq_len, silu_scale=silu_scale,
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, mask, attn_bias = ctx.saved_tensors
        q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_dense_backward(
            grad_output, q, k, v, mask, attn_bias,
            ctx.mask_type, ctx.max_seq_len, ctx.silu_scale,
        )
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
    """jagged 布局 hstu_jagged.equal 的自动微分包装（q/k 等长）。"""

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
        return ops_rec.ascendc.attention.hstu_jagged.equal(
            q, k, v, mask=mask, attn_bias=attn_bias, mask_type=mask_type,
            max_seq_len=max_seq_len, silu_scale=silu_scale, seq_offset=seq_offset,
            num_context=num_context, num_target=num_target,
            target_group_size=target_group_size, alpha=alpha,
        )

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, mask, attn_bias, seq_offset, num_context, num_target = ctx.saved_tensors
        q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_jagged_backward(
            grad_output, q, k, v, mask, attn_bias, ctx.mask_type,
            ctx.max_seq_len, ctx.max_seq_len, ctx.silu_scale,
            seq_offset, seq_offset, num_context, num_target,
            ctx.target_group_size, ctx.alpha,
        )
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def generate_dense_tensor(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    grad = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    q = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    k = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    v = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == MaskType.TRIL:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == MaskType.NONE:
        mask = None
    else:
        mask = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    return grad, q, k, v, bias, mask


def generate_jagged_tensor(batch_size, max_seq_len, head_num_q, head_dim_qk, head_dim_v, mask_type, data_type):
    """生成变长序列数据（equal 语义：q/k 严格等长，共用 seq_offset）。"""
    min_seq_len = 1
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens, axis=0)))
    total_len = torch.sum(seq_lens).item()

    grad = torch.rand(total_len, head_num_q, head_dim_v, dtype=data_type).uniform_(-1, 1)
    q = torch.rand(total_len, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    k = torch.rand(total_len, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(total_len, head_num_q, head_dim_v, dtype=data_type).uniform_(-1, 1)

    bias = torch.rand(batch_size, head_num_q, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == MaskType.TRIL:
        mask = torch.tril(torch.ones(batch_size, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == MaskType.NONE:
        mask = None
    else:
        mask = torch.rand(batch_size, head_num_q, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    return grad, q, k, v, bias, mask, seq_offset, seq_offset


def check_grads(grads, grads_ref, label, loss):
    """逐个对比 autograd 梯度与直接调用 backward 算子的输出。"""
    names = ("q", "k", "v", "attn_bias")
    assert len(grads) == len(grads_ref) == len(names)
    for name, grad, grad_ref in zip(names, grads, grads_ref):
        if grad_ref is None:
            assert grad is None, f"{label}: {name} 不应有梯度"
            continue
        assert grad is not None, f"{label}: {name} 缺少梯度"
        assert allclose(grad, grad_ref, loss, 1e-4), f"{label}: {name} 梯度不一致"


class TestHstuAutograd:
    # ────────────────── dense ──────────────────

    def execute_dense(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale,
                      enable_bias, data_type):
        grad, q, k, v, bias, mask = generate_dense_tensor(
            batch_size, max_seq_len, head_num, head_dim, mask_type, data_type
        )
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = bias.to("npu") if enable_bias else None
        mask_npu = mask.to("npu") if mask_type in (MaskType.TRIL, MaskType.CUSTOM) else None

        # 参照：直接调用 backward 算子
        grads_ref = ops_rec.ascendc.attention.hstu_dense_backward(
            grad_npu, q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len, silu_scale
        )
        if not enable_bias:
            grads_ref = grads_ref[:3] + (None,)

        # 训练链路：y = autograd(...) -> loss = sum(y * grad) -> loss.backward()
        q_ = q_npu.detach().clone().requires_grad_(True)
        k_ = k_npu.detach().clone().requires_grad_(True)
        v_ = v_npu.detach().clone().requires_grad_(True)
        b_ = bias_npu.detach().clone().requires_grad_(True) if enable_bias else None

        y = HstuDenseAutograd.apply(q_, k_, v_, mask_npu, b_, mask_type, max_seq_len, silu_scale)
        (y * grad_npu).sum().backward()
        torch.npu.synchronize()

        grads = (q_.grad, k_.grad, v_.grad, b_.grad if b_ is not None else None)
        loss_tol = 1e-4 if data_type == torch.float32 else 5e-3
        check_grads(grads, grads_ref, "dense autograd", loss_tol)

    @pytest.mark.parametrize("batch_size", [1, 4])  # 范围: [1, 2048]
    @pytest.mark.parametrize("max_seq_len", [64, 512])  # 范围: [1, 20480]
    @pytest.mark.parametrize("head_num", [4])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_dim", [32])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16])
    def test_dense_autograd(self, batch_size, max_seq_len, head_num, head_dim, mask_type,
                            silu_scale, enable_bias, data_type):
        """eager：y.backward() 梯度与直接调用 hstu_dense_backward 一致。"""
        self.execute_dense(batch_size, max_seq_len, head_num, head_dim, mask_type,
                           silu_scale, enable_bias, data_type)

    @pytest.mark.graph
    def test_dense_autograd_graph(self):
        """graph：Python 侧 autograd.Function 包装可被 dynamo 追踪（forward 走 Meta 构图）。"""
        B, S, N, D = 1, 16, 4, 32
        data_type = torch.float16
        q = torch.rand(B, S, N, D, device="npu", dtype=data_type).uniform_(-1, 1)
        k = torch.rand(B, S, N, D, device="npu", dtype=data_type).uniform_(-1, 1)
        v = torch.rand(B, S, N, D, device="npu", dtype=data_type).uniform_(-1, 1)
        grad = torch.rand(B, S, N, D, device="npu", dtype=data_type).uniform_(-1, 1)

        def run(forward_fn):
            q_ = q.detach().clone().requires_grad_(True)
            k_ = k.detach().clone().requires_grad_(True)
            v_ = v.detach().clone().requires_grad_(True)
            y = forward_fn(q_, k_, v_)
            (y * grad).sum().backward()
            return y, q_.grad, k_.grad, v_.grad

        compiled_fn = torch_compiled(
            lambda q_, k_, v_: HstuDenseAutograd.apply(q_, k_, v_, None, None, 0, S, 1.0 / S),
            dynamic=True,
        )
        y_eager, qg_eager, kg_eager, vg_eager = run(
            lambda q_, k_, v_: HstuDenseAutograd.apply(q_, k_, v_, None, None, 0, S, 1.0 / S)
        )
        y_compiled, qg_compiled, kg_compiled, vg_compiled = run(compiled_fn)
        torch.npu.synchronize()

        for name, eager, compiled in (("y", y_eager, y_compiled),
                                      ("q_grad", qg_eager, qg_compiled),
                                      ("k_grad", kg_eager, kg_compiled),
                                      ("v_grad", vg_eager, vg_compiled)):
            assert allclose(eager, compiled, 5e-3, 1e-4), f"dense autograd graph: {name} 不一致"

    # ────────────────── jagged（equal）──────────────────

    def execute_jagged(self, batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v,
                       mask_type, silu_scale, enable_bias, data_type):
        grad, q, k, v, bias, mask, seq_offset_q, seq_offset_k = generate_jagged_tensor(
            batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v, mask_type, data_type
        )
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = bias.to("npu") if enable_bias else None
        mask_npu = mask.to("npu") if mask_type in (MaskType.TRIL, MaskType.CUSTOM) else None
        seq_offset_q = seq_offset_q.to("npu")

        # 参照：直接调用 backward 算子（equal 语义：q/k 等长，两个 offset 相同）
        grads_ref = ops_rec.ascendc.attention.hstu_jagged_backward(
            grad_npu, q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type,
            max_seq_len, max_seq_len, silu_scale, seq_offset_q, seq_offset_q,
            None, None, None, 1.0,
        )
        if not enable_bias:
            grads_ref = grads_ref[:3] + (None,)

        q_ = q_npu.detach().clone().requires_grad_(True)
        k_ = k_npu.detach().clone().requires_grad_(True)
        v_ = v_npu.detach().clone().requires_grad_(True)
        b_ = bias_npu.detach().clone().requires_grad_(True) if enable_bias else None

        y = HstuJaggedAutograd.apply(
            q_, k_, v_, mask_npu, b_, mask_type, max_seq_len, silu_scale,
            seq_offset_q, None, None, None, 1.0,
        )
        (y * grad_npu).sum().backward()
        torch.npu.synchronize()

        grads = (q_.grad, k_.grad, v_.grad, b_.grad if b_ is not None else None)
        loss_tol = 1e-4 if data_type == torch.float32 else 5e-3
        check_grads(grads, grads_ref, "jagged autograd", loss_tol)

    @pytest.mark.parametrize("batch_size", [1, 4])  # 范围: [1, 2048]
    @pytest.mark.parametrize("max_seq_len", [64, 512])  # 范围: [1, 20480]
    @pytest.mark.parametrize("head_num", [4])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_dim_qk", [32])  # 范围: [1, 512]
    @pytest.mark.parametrize("head_dim_v", [32])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16])
    def test_jagged_autograd(self, batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v,
                             mask_type, silu_scale, enable_bias, data_type):
        """eager：y.backward() 梯度与直接调用 hstu_jagged_backward 一致。"""
        self.execute_jagged(batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v,
                            mask_type, silu_scale, enable_bias, data_type)

    @pytest.mark.graph
    def test_jagged_autograd_graph(self):
        """graph：jagged 包装同样可被 dynamo 追踪，eager vs compiled 梯度一致。"""
        batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v = 2, 64, 4, 32, 32
        data_type = torch.float16
        grad, q, k, v, bias, mask, seq_offset_q, seq_offset_k = generate_jagged_tensor(
            batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v, MaskType.TRIL, data_type
        )
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        mask_npu = mask.to("npu")
        seq_offset_q = seq_offset_q.to("npu")

        def run(forward_fn):
            q_ = q_npu.detach().clone().requires_grad_(True)
            k_ = k_npu.detach().clone().requires_grad_(True)
            v_ = v_npu.detach().clone().requires_grad_(True)
            y = forward_fn(q_, k_, v_)
            (y * grad_npu).sum().backward()
            return y, q_.grad, k_.grad, v_.grad

        compiled_fn = torch_compiled(
            lambda q_, k_, v_: HstuJaggedAutograd.apply(
                q_, k_, v_, mask_npu, None, MaskType.TRIL, max_seq_len, 1.0 / max_seq_len,
                seq_offset_q, None, None, None, 1.0,
            ),
            dynamic=True,
        )
        y_eager, qg_eager, kg_eager, vg_eager = run(
            lambda q_, k_, v_: HstuJaggedAutograd.apply(
                q_, k_, v_, mask_npu, None, MaskType.TRIL, max_seq_len, 1.0 / max_seq_len,
                seq_offset_q, None, None, None, 1.0,
            )
        )
        y_compiled, qg_compiled, kg_compiled, vg_compiled = run(compiled_fn)
        torch.npu.synchronize()

        for name, eager, compiled in (("y", y_eager, y_compiled),
                                      ("q_grad", qg_eager, qg_compiled),
                                      ("k_grad", kg_eager, kg_compiled),
                                      ("v_grad", vg_eager, vg_compiled)):
            assert allclose(eager, compiled, 5e-3, 1e-4), f"jagged autograd graph: {name} 不一致"


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
