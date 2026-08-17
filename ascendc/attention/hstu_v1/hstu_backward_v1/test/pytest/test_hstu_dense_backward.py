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
import sys

import pytest
import torch
import torch.nn.functional as F

import ops_rec

from test_common_utils import allclose

# pylint: disable=duplicate-code
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    grad = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    q = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    k = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    v = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    return grad, q, k, v, bias, mask


class TestHstuNormalDemo:
    @staticmethod
    def golden_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
    ):
        batch_size, seq_len, head_num, head_dim = grad.shape

        qk = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        gv = torch.matmul(grad.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))

        qk = qk.float()
        gv = gv.float()

        if mask_type in (0, 3):
            mask = mask.float()

        if enable_bias:
            bias = bias.float()
            qkb = qk + bias
        else:
            qkb = qk

        real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

        if mask_type in (0, 3):
            score = F.silu(qkb) * real_silu_scale * mask
        else:
            score = F.silu(qkb) * real_silu_scale

        score = score.to(data_type)
        v_grad = torch.matmul(score.permute(0, 1, 3, 2), grad.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        if mask_type in (0, 3):
            attn_bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        else:
            attn_bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))

        attn_bias_grad = attn_bias_grad.to(data_type)
        k_grad = torch.matmul(attn_bias_grad.permute(0, 1, 3, 2), q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        q_grad = torch.matmul(attn_bias_grad, k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), attn_bias_grad.cpu()

    @staticmethod
    def custom_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
    ):
        if enable_bias:
            q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_dense_backward(
                grad, q, k, v, mask, bias, mask_type, max_seq_len, silu_scale
            )
        else:
            q_grad, k_grad, v_grad, attn_bias_grad = ops_rec.ascendc.attention.hstu_dense_backward(
                grad, q, k, v, mask, None, mask_type, max_seq_len, silu_scale
            )

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), attn_bias_grad.cpu()

    def execute(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        grad, q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)

        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = None
        if enable_bias:
            bias_npu = bias.to("npu")
        mask_npu = None
        if mask_type in (0, 3):
            mask_npu = mask.to("npu")

        q_grad, k_grad, v_grad, attn_bias_grad = self.custom_op_exec(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            bias_npu,
            mask_npu,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
        )

        q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = self.golden_op_exec(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            bias_npu,
            mask_npu,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
        )

        torch.npu.synchronize()

        loss = 1e-4
        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 5e-3

        q_res = allclose(q_grad, q_grad_golden, loss, loss)
        k_res = allclose(k_grad, k_grad_golden, loss, loss)
        v_res = allclose(v_grad, v_grad_golden, loss, loss)
        bias_res = not enable_bias or allclose(attn_bias_grad, attn_bias_grad_golden, loss, loss)

        assert q_res and k_res and v_res and bias_res

    @pytest.mark.parametrize("batch_size", [1, 4])  # 范围: [1, 2048]
    @pytest.mark.parametrize("max_seq_len", [1, 257, 512, 1234])  # 范围: [1, 20480]
    @pytest.mark.parametrize("head_num", [1, 16])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_dim", [16, 32])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_normal_case(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        self.execute(
            batch_size,
            max_seq_len,
            head_num,
            head_dim,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
        )

    @pytest.mark.ci
    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("max_seq_len", [128, 512])
    @pytest.mark.parametrize("head_num", [4])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float32])
    def test_hstu_normal_ci(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        """CI 快速冒烟用例：固定 shape，仅验证正确性"""
        self.execute(batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)

    @pytest.mark.graph
    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("max_seq_len", [64])
    @pytest.mark.parametrize("head_num", [4])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("mask_type", [2])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [False])
    @pytest.mark.parametrize("data_type", [torch.float32])
    def test_hstu_normal_graph(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        """Graph 模式用例：验证 torch.compile 后端正确性"""
        grad, q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)
        grad = grad.npu()
        q = q.npu()
        k = k.npu()
        v = v.npu()
        if mask is not None:
            mask = mask.npu()

        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.hstu_dense_backward,
            dynamic=True,
            fullgraph=True,
        )
        q_grad, k_grad, v_grad, _ = compiled_fn(
            grad, q, k, v, mask, None, mask_type=mask_type, max_seq_len=max_seq_len, silu_scale=silu_scale
        )

        q_grad_golden, k_grad_golden, v_grad_golden, _ = self.golden_op_exec(
            grad, q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type
        )

        loss = 1e-4
        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 5e-3

        q_res = allclose(q_grad, q_grad_golden, loss, loss)
        k_res = allclose(k_grad, k_grad_golden, loss, loss)
        v_res = allclose(v_grad, v_grad_golden, loss, loss)
        assert q_res and k_res and v_res, "hstu_dense_backward graph mode allclose check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
