#!/usr/bin/env python3
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
hstu_v1 训练自动微分（autograd）使用样例

背景:
    hstu_v1 前向算子（hstu_dense / hstu_jagged.equal|delta）在 torch_extension 中不注册
    C++ AutogradPrivateUse1——autograd Function 黑盒会导致 inductor 图模式数值错误，
    图模式仅走 PrivateUse1 + Meta 干净路径（与 paged/ln_mul/disentangle_attention
    已验证模式一致）。反向梯度由独立算子 hstu_dense_backward / hstu_jagged_backward 提供。

    如需训练自动微分（loss.backward()），请在 Python 侧用 torch.autograd.Function 包装：
        forward  -> 调用前向算子
        backward -> 调用对应的 backward 算子
    本样例演示 dense 与 jagged（equal）两种布局的包装方法，并校验 y.backward() 得到的
    q/k/v/bias 梯度与直接调用 backward 算子结果一致。

用法:
    python3 test_torch_hstu_backward_v1_autograd.py
"""

import random

import numpy as np
import torch
import torch_npu

import ops_rec

# pylint: disable=duplicate-code
torch.npu.config.allow_internal_format = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化


# ---------------------------------------------------------------------------
# Python 侧自动微分包装（训练场景核心用法，可直接复制到训练代码中）
# ---------------------------------------------------------------------------

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
        # 返回值顺序与 forward 入参一一对应：q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale
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
    """jagged 布局 hstu_jagged.equal 的自动微分包装（q/k 等长，seq_offset 共用）。

    forward 调用 ops_rec.ascendc.attention.hstu_jagged.equal，
    backward 调用 ops_rec.ascendc.attention.hstu_jagged_backward（equal 语义：
    max_seq_len_k = max_seq_len，seq_offset_k = seq_offset）。
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
        return ops_rec.ascendc.attention.hstu_jagged.equal(
            q,
            k,
            v,
            mask=mask,
            attn_bias=attn_bias,
            mask_type=mask_type,
            max_seq_len=max_seq_len,
            silu_scale=silu_scale,
            seq_offset=seq_offset,
            num_context=num_context,
            num_target=num_target,
            target_group_size=target_group_size,
            alpha=alpha,
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


# ---------------------------------------------------------------------------
# 校验：y.backward() 梯度 vs 直接调用 backward 算子
# ---------------------------------------------------------------------------

def _check_grads(label, grads, grads_ref):
    """逐个对比 autograd 梯度与直接调用 backward 算子的输出。"""
    assert len(grads) == len(grads_ref)
    for name, grad, grad_ref in zip(("q", "k", "v", "attn_bias"), grads, grads_ref):
        if grad_ref is None:
            assert grad is None, f"{label}: {name} 不应有梯度"
            continue
        assert grad is not None, f"{label}: {name} 缺少梯度"
        ok = torch.allclose(grad, grad_ref, rtol=1e-4, atol=1e-4)
        print(f"  {name}_grad 一致: {ok}")
        assert ok, f"{label}: {name} 梯度不一致"


def check_dense_autograd(mask_type, enable_bias, silu_scale=0.0):
    batch_size, max_seq_len, head_num, head_dim = 2, 64, 4, 32
    dtype = torch.float32

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype)).npu()
    elif mask_type == 3:
        mask = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()
    else:  # mask_type == 2
        mask = None

    bias = None
    if enable_bias:
        bias = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()

    grad = torch.empty(batch_size, max_seq_len, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    q = torch.empty(batch_size, max_seq_len, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    k = torch.empty(batch_size, max_seq_len, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()
    v = torch.empty(batch_size, max_seq_len, head_num, head_dim, dtype=dtype).uniform_(-1, 1).npu()

    # 参照：直接调用 backward 算子
    grads_ref = ops_rec.ascendc.attention.hstu_dense_backward(
        grad, q, k, v, mask, bias, mask_type, max_seq_len, silu_scale
    )
    if bias is None:
        grads_ref = grads_ref[:3] + (None,)

    # 训练链路：y = autograd(q, k, v, ...) -> loss = sum(y * grad) -> loss.backward()
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    b_ = bias.detach().clone().requires_grad_(True) if bias is not None else None

    y = HstuDenseAutograd.apply(q_, k_, v_, mask, b_, mask_type, max_seq_len, silu_scale)
    loss = (y * grad).sum()
    loss.backward()
    torch.npu.synchronize()

    grads = (q_.grad, k_.grad, v_.grad, b_.grad if b_ is not None else None)
    print(f"[dense] hstu_dense 自动微分 (mask_type={mask_type}, enable_bias={enable_bias}):")
    _check_grads("dense", grads, grads_ref)
    print()


def _generate_jagged_qkv(batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v, dtype):
    """生成变长序列数据，返回 q/k/v 及 seq_offset（equal 语义：q/k 严格等长，共用 seq_offset）。"""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens, axis=0)))
    total_len = torch.sum(seq_lens).item()

    q = torch.rand(total_len, head_num, head_dim_qk, dtype=dtype).uniform_(-1, 1).npu()
    k = torch.rand(total_len, head_num, head_dim_qk, dtype=dtype).uniform_(-1, 1).npu()
    v = torch.rand(total_len, head_num, head_dim_v, dtype=dtype).uniform_(-1, 1).npu()

    return q, k, v, seq_offset.npu(), seq_offset.npu()


def check_jagged_autograd(mask_type, enable_bias, silu_scale=0.0):
    batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v = 2, 64, 4, 32, 32
    dtype = torch.float32
    alpha = 1.0

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, max_seq_len, max_seq_len, dtype=dtype)).npu()
    elif mask_type == 3:
        mask = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()
    else:  # mask_type == 2
        mask = None

    bias = None
    if enable_bias:
        bias = torch.empty(batch_size, head_num, max_seq_len, max_seq_len, dtype=dtype).uniform_(-1, 1).npu()

    q, k, v, seq_offset, _ = _generate_jagged_qkv(
        batch_size, max_seq_len, head_num, head_dim_qk, head_dim_v, dtype
    )
    grad = torch.rand(q.shape[0], head_num, head_dim_v, dtype=dtype).uniform_(-1, 1).npu()

    # 参照：直接调用 backward 算子（equal 语义：max_seq_len_k = max_seq_len，seq_offset_k = seq_offset）
    grads_ref = ops_rec.ascendc.attention.hstu_jagged_backward(
        grad, q, k, v, mask, bias, mask_type, max_seq_len, max_seq_len, silu_scale,
        seq_offset, seq_offset, None, None, None, alpha,
    )
    if bias is None:
        grads_ref = grads_ref[:3] + (None,)

    # 训练链路
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    b_ = bias.detach().clone().requires_grad_(True) if bias is not None else None

    y = HstuJaggedAutograd.apply(
        q_, k_, v_, mask, b_, mask_type, max_seq_len, silu_scale,
        seq_offset, None, None, None, alpha,
    )
    loss = (y * grad).sum()
    loss.backward()
    torch.npu.synchronize()

    grads = (q_.grad, k_.grad, v_.grad, b_.grad if b_ is not None else None)
    print(f"[jagged] hstu_jagged.equal 自动微分 (mask_type={mask_type}, enable_bias={enable_bias}):")
    _check_grads("jagged", grads, grads_ref)
    print()


def main():
    torch.npu.set_device(0)
    set_seed(1234)
    for mask_type, enable_bias in (
        (0, False),
        (0, True),
        (2, False),
        (2, True),
        (3, True),
    ):
        check_dense_autograd(mask_type, enable_bias)
        check_jagged_autograd(mask_type, enable_bias)
    print("hstu_v1 自动微分示例全部执行完成。")


if __name__ == "__main__":
    main()
