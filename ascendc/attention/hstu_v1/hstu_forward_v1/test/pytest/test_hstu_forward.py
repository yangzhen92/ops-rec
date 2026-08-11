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

import sys
import pytest
import torch
import numpy as np

import ops_rec

torch.npu.config.allow_internal_format = False


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def get_golden(q, k, v, attn_bias=None, mask=None, mask_type=0, silu_scale=None, max_seq_len=None):
    """numpy 实现 HSTU 前向：out = (silu(qk^T + bias) * silu_scale * mask) @ v（fp32 精度）"""
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    B, S, N, D = q.shape
    scale = silu_scale if silu_scale not in (None, 0.0) else 1.0 / max_seq_len

    qk = np.einsum("bnhd,bmhd->bhnm", q_f, k_f)
    if attn_bias is not None:
        qk = qk + attn_bias.astype(np.float32)
    qk = qk * (1.0 / (1.0 + np.exp(-qk))) * scale  # silu
    if mask is not None:
        qk = qk * mask.astype(np.float32)
    elif mask_type == 0:
        qk = qk * np.tril(np.ones((S, S), dtype=np.float32))[None, None, :, :]
    out = np.einsum("bhnm,bmhd->bnhd", qk, v_f)
    return out.astype(q.dtype)


def get_golden_jagged(q, k, v, seq_offset, silu_scale, mask_type=0):
    """numpy 实现 HSTU jagged（equal）前向：按 batch 拆分后每段做 silu(qk^T)*scale @ v"""
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    outs = []
    for b in range(len(seq_offset) - 1):
        s0, s1 = seq_offset[b], seq_offset[b + 1]
        qb, kb, vb = q_f[s0:s1], k_f[s0:s1], v_f[s0:s1]
        S = qb.shape[0]
        qk = np.einsum("nhd,mhd->hnm", qb, kb)
        qk = qk * (1.0 / (1.0 + np.exp(-qk))) * silu_scale  # silu
        if mask_type == 0:
            qk = qk * np.tril(np.ones((S, S), dtype=np.float32))[None, :, :]
        outs.append(np.einsum("hnm,mhd->nhd", qk, vb))
    return np.concatenate(outs, axis=0).astype(q.dtype)


def get_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 2e-2, 2e-2  # fp16/bf16 累加误差放宽


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32), (2, 32, 4, 32)])
def test_hstu_dense_eager(dtype, shape):
    """eager 模式：MHA，mask_type=0 内置下三角"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask_type=0, silu_scale=1.0 / S, max_seq_len=S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_gqa_dimmismatch(dtype, shape):
    """GQA（N_k < N_q）且 qk/v dim 不等"""
    torch.npu.set_device(0)
    B, S, N_q, D_qk = shape
    N_k, D_v = 2, 64
    q = torch.randn(B, S, N_q, D_qk, device="npu", dtype=dtype)
    k = torch.randn(B, S, N_k, D_qk, device="npu", dtype=dtype)
    v = torch.randn(B, S, N_k, D_v, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask_type=0, silu_scale=1.0 / S, max_seq_len=S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)

    assert y_op.shape == (B, S, N_q, D_v)
    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_custom_mask(dtype, shape):
    """mask_type=3：自定义 mask（下三角 0/1）"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)
    tril = np.tril(np.ones((S, S), dtype=np.float32))
    mask = torch.from_numpy(np.broadcast_to(tril, (B, N, S, S))).to(device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=mask.cpu().numpy(), mask_type=3, silu_scale=1.0 / S, max_seq_len=S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask, None, 3, S, 1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_graph(dtype, shape):
    """inductor 图模式（Meta 注册走通构图）"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype)
    k = torch.randn(*shape, device="npu", dtype=dtype)
    v = torch.randn(*shape, device="npu", dtype=dtype)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask_type=0, silu_scale=1.0 / S, max_seq_len=S)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_dense, dynamic=True, fullgraph=True)
    y_op = compiled_fn(q=q, k=k, v=v, mask=None, attn_bias=None,
                       mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
def test_hstu_dense_backward_meta():
    """inductor 构图：backward 算子 Meta 推导（fake tensor 下 shape 与 eager 实现一致）"""
    B, S, N, D = 1, 16, 4, 32
    torch.npu.set_device(0)
    grad = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    q = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    with torch._subclasses.fake_tensor.FakeTensorMode():
        gq, gk, gv, gbias = torch.ops.ops_rec.hstu_dense_backward(grad, q, k, v, None, None, 0, S, 1.0 / S)
        assert gq.shape == q.shape and gq.dtype == q.dtype
        assert gk.shape == k.shape and gv.shape == v.shape
        # 无 attn_bias 时 bias 梯度为 256 对齐 shape（与 eager 实现一致）
        assert gbias.shape == (B, N, (S + 255) // 256 * 256, (S + 255) // 256 * 256)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("shape", [(1, 16, 4, 32)])
def test_hstu_dense_backward_graph(dtype, shape):
    """inductor 图模式：训练闭环（forward 编译 + 梯度回传），梯度与 eager 一致"""
    torch.npu.set_device(0)
    B, S, N, D = shape
    q = torch.randn(*shape, device="npu", dtype=dtype, requires_grad=True)
    k = torch.randn(*shape, device="npu", dtype=dtype, requires_grad=True)
    v = torch.randn(*shape, device="npu", dtype=dtype, requires_grad=True)

    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_dense, dynamic=True, fullgraph=True)
    y = compiled_fn(q, k, v, None, None, 0, S, 1.0 / S)
    y.sum().backward()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    y2 = ops_rec.ascendc.attention.hstu_dense(q2, k2, v2, None, None, 0, S, 1.0 / S)
    y2.sum().backward()

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(q.grad.cpu().numpy(), q2.grad.cpu().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(k.grad.cpu().numpy(), k2.grad.cpu().numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(v.grad.cpu().numpy(), v2.grad.cpu().numpy(), rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float16])
def test_hstu_jagged_graph(dtype):
    """inductor 图模式：jagged（equal）前向入图"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    q = torch.randn(total, N, D, device="npu", dtype=dtype)
    k = torch.randn(total, N, D, device="npu", dtype=dtype)
    v = torch.randn(total, N, D, device="npu", dtype=dtype)
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y_golden = get_golden_jagged(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                                 seq_offset.cpu().numpy(), silu_scale=1.0 / S, mask_type=0)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_jagged.equal, dynamic=True, fullgraph=True)
    y_op = compiled_fn(q, k, v, None, None, 0, S, 1.0 / S, seq_offset)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.ci
def test_hstu_dense_ci():
    """CI 快速冒烟用例：固定 shape + float16，仅验证正确性"""
    torch.npu.set_device(0)
    B, S, N, D = 1, 16, 4, 32
    q = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask_type=0, silu_scale=1.0 / S, max_seq_len=S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)

    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
