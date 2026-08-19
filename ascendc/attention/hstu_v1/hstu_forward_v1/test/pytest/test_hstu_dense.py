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
# HSTU dense 前向测试，迁移自 RecSDK test_hstu_dense_forward.py
# ==============================================================================

import pytest
import torch
import torch.nn.functional as F
import numpy as np

import ops_rec
from common import MaskType, allclose, get_chip, get_golden, get_tolerance, torch_compiled


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    q = torch.rand(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    k = torch.rand(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    v = torch.rand(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len).uniform_(-1, 1)
    if get_chip():
        invalid_attn_mask = torch.randint(0, 2, (max_seq_len, max_seq_len))
        invalid_attn_mask = torch.tril(invalid_attn_mask)
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elif mask_type == MaskType.TRIL:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    return (q.to(data_type).to("npu"),
            k.to(data_type).to("npu"),
            v.to(data_type).to("npu"),
            rel_attn_bias.to(data_type).to("npu"),
            invalid_attn_mask.to(data_type).to("npu"))


class TestHstuDenseDemo:
    @staticmethod
    def golden_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        b, n, num_heads, linear_dim = q.shape
        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q, k)

        qk_attn = qk_attn.to(torch.float32)
        bias = bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if get_chip():
            mask = mask.repeat(1, num_heads, 1, 1)
            qk_attn = qk_attn * mask
        elif mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        v = v.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v)
        attn_output = attn_output.permute(0, 2, 1, 3)
        torch.npu.synchronize()
        return attn_output.cpu().to(data_type).reshape(-1)

    @staticmethod
    def custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        if enable_bias:
            output = ops_rec.ascendc.attention.hstu_dense(
                q, k, v, mask=mask, attn_bias=bias, mask_type=mask_type,
                max_seq_len=max_seq_len, silu_scale=silu_scale
            )
        else:
            output = ops_rec.ascendc.attention.hstu_dense(
                q, k, v, mask=mask, attn_bias=None, mask_type=mask_type,
                max_seq_len=max_seq_len, silu_scale=silu_scale
            )

        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, data_type, mask_type)

        torch.npu.synchronize()

        output = self.custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)
        golden = self.golden_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)

        torch.npu.synchronize()

        if data_type == torch.bfloat16:
            res = allclose(output, golden, 5e-3, 5e-3)
        elif data_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [1, 4])
    @pytest.mark.parametrize("max_seq_len", [15,256, 1024, 4095])
    @pytest.mark.parametrize("head_dim", [32, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dense_forward(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)


# ────────────────── Inductor 图模式 ──────────────────

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
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    compiled_fn = torch_compiled(ops_rec.ascendc.attention.hstu_dense, dynamic=True, fullgraph=True,
                                 backend="inductor")
    y_op = compiled_fn(q=q, k=k, v=v, mask=None, attn_bias=None,
                       mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    rtol, atol = get_tolerance(dtype)
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


# ────────────────── CI 冒烟测试 ──────────────────

@pytest.mark.ci
def test_hstu_dense_ci():
    """CI 快速冒烟用例：固定 shape + float16，仅验证正确性"""
    torch.npu.set_device(0)
    B, S, N, D = 1, 16, 4, 32
    q = torch.rand(B, S, N, D, device="npu", dtype=torch.float16).uniform_(-1, 1)
    k = torch.rand(B, S, N, D, device="npu", dtype=torch.float16).uniform_(-1, 1)
    v = torch.rand(B, S, N, D, device="npu", dtype=torch.float16).uniform_(-1, 1)

    y_golden = get_golden(q.cpu().numpy(), k.cpu().numpy(), v.cpu().numpy(),
                          mask=None, attn_bias=None, mask_type=0,
                          max_seq_len=S, silu_scale=1.0 / S)
    y_op = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None,
                                                mask_type=0, max_seq_len=S, silu_scale=1.0 / S)

    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
