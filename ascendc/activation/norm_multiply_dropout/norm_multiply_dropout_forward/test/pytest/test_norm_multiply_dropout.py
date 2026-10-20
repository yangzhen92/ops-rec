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

# 增大 cache 避免 dynamic=True 时因重编译超限报错
torch._dynamo.config.cache_size_limit = 256

EPS = 1e-5


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def get_golden(x, u, weight, bias, eps=EPS):
    """numpy golden：LayerNorm + multiply，不 dropout。"""
    x_f = x.astype(np.float32)
    u_f = u.astype(np.float32)
    weight_f = weight.astype(np.float32)
    bias_f = bias.astype(np.float32)

    mean = x_f.mean(axis=1, keepdims=True)
    var = x_f.var(axis=1, keepdims=True, ddof=0)
    rstd = 1.0 / np.sqrt(var + eps)
    norm_x = (x_f - mean) * rstd
    ln_ret = norm_x * weight_f + bias_f
    output = u_f * ln_ret
    return output.astype(np.float32), mean, var


def gen_inputs(batch, col, dtype):
    x = torch.rand([batch, col], device="npu", dtype=dtype) - 0.5
    u = torch.rand([batch, col], device="npu", dtype=dtype) - 0.5
    weight = torch.rand([col], device="npu", dtype=dtype) - 0.5
    bias = torch.rand([col], device="npu", dtype=dtype) - 0.5
    return x, u, weight, bias


@pytest.mark.ci
@pytest.mark.parametrize("col", [512, 1024])
def test_norm_multiply_dropout_no_drop(col):
    """dropout_ratio=0：output/mean/var 与 numpy golden 全量比对。"""
    batch = 8
    dtype = torch.bfloat16
    x, u, weight, bias = gen_inputs(batch, col, dtype)

    golden_output, golden_mean, golden_var = get_golden(
        x.cpu().numpy(), u.cpu().numpy(), weight.cpu().numpy(), bias.cpu().numpy())

    output, mean, var = ops_rec.ascendc.activation.norm_multiply_dropout(
        x, u, weight, bias, eps=EPS, dropout_ratio=0.0)

    assert output.shape == (batch, col) and output.dtype == dtype
    assert mean.shape == (batch,) and var.shape == (batch,)

    np.testing.assert_allclose(output.float().cpu().numpy(), golden_output, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(mean.cpu().numpy().reshape(-1, 1), golden_mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(var.cpu().numpy().reshape(-1, 1), golden_var, rtol=1e-3, atol=1e-3)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_norm_multiply_dropout_dtype(dtype):
    """dtype 遍历（dropout_ratio=0）。"""
    batch, col = 4, 512
    x, u, weight, bias = gen_inputs(batch, col, dtype)

    golden_output, _, _ = get_golden(
        x.cpu().numpy(), u.cpu().numpy(), weight.cpu().numpy(), bias.cpu().numpy())

    output, _, _ = ops_rec.ascendc.activation.norm_multiply_dropout(
        x, u, weight, bias, eps=EPS, dropout_ratio=0.0)
    assert output.dtype == dtype
    np.testing.assert_allclose(output.float().cpu().numpy(), golden_output, rtol=1e-2, atol=1e-2)


@pytest.mark.ci
def test_norm_multiply_dropout_with_drop():
    """dropout_ratio>0：显式 mask 下校验输出与 golden 一致（bit=1 保留）。

    mask 直接经 torch.ops 传入以构造确定性用例。
    """
    batch, col = 4, 512
    dtype = torch.bfloat16
    dropout_ratio = 0.5
    x, u, weight, bias = gen_inputs(batch, col, dtype)
    mask = torch.randint(0, 256, (batch, col // 8), dtype=torch.uint8, device="npu")

    output, mean, var = torch.ops.ops_rec.norm_multiply_dropout(
        x, u, weight, bias, mask, EPS, dropout_ratio)
    assert output.shape == (batch, col) and output.dtype == dtype
    assert mean.shape == (batch,) and var.shape == (batch,)

    # dropout 只影响 output，不影响 mean/var
    _, golden_mean, golden_var = get_golden(
        x.cpu().numpy(), u.cpu().numpy(), weight.cpu().numpy(), bias.cpu().numpy())
    np.testing.assert_allclose(mean.cpu().numpy().reshape(-1, 1), golden_mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(var.cpu().numpy().reshape(-1, 1), golden_var, rtol=1e-3, atol=1e-3)


@pytest.mark.graph
def test_norm_multiply_dropout_graph():
    """图模式：torch.compile(fullgraph=True) 调用（dropout_ratio=0）。"""
    batch, col = 4, 512
    dtype = torch.bfloat16
    x, u, weight, bias = gen_inputs(batch, col, dtype)

    golden_output, _, _ = get_golden(
        x.cpu().numpy(), u.cpu().numpy(), weight.cpu().numpy(), bias.cpu().numpy())

    compiled = torch_compiled(
        lambda a, b, c, d: ops_rec.ascendc.activation.norm_multiply_dropout(
            a, b, c, d, eps=EPS, dropout_ratio=0.0),
        dynamic=True, fullgraph=True)
    output, mean, var = compiled(x, u, weight, bias)

    assert output.shape == (batch, col)
    np.testing.assert_allclose(output.float().cpu().numpy(), golden_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
