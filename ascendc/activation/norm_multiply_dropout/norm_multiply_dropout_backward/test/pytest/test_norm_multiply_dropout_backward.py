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
"""norm_multiply_dropout autograd 数值梯度比对测试（dropout_ratio=0）。"""

import pytest
import torch
import numpy as np

import ops_rec

torch.npu.config.allow_internal_format = False

EPS = 1e-5
RTOL = 1e-2
ATOL = 1e-2


def get_golden_backward(x, u, weight, bias, eps=EPS):
    """numpy golden：LayerNorm + multiply 反向。"""
    x_f = x.astype(np.float32)
    u_f = u.astype(np.float32)
    weight_f = weight.astype(np.float32)
    bias_f = bias.astype(np.float32)

    mean = x_f.mean(axis=1, keepdims=True)
    var = x_f.var(axis=1, keepdims=True, ddof=0)
    rstd = 1.0 / np.sqrt(var + eps)
    norm_x = (x_f - mean) * rstd
    ln_ret = norm_x * weight_f + bias_f

    batch = x_f.shape[0]
    d_out = np.ones_like(ln_ret)
    d_u = d_out * ln_ret
    d_ln_ret = d_out * u_f
    d_weight = (d_ln_ret * norm_x).sum(axis=0, keepdims=False)
    d_bias = d_ln_ret.sum(axis=0, keepdims=False)
    d_norm_x = d_ln_ret * weight_f
    d_var = (d_norm_x * (x_f - mean) * (-0.5) * rstd ** 3).sum(axis=1, keepdims=True)
    d_mean = (d_norm_x * (-rstd)).sum(axis=1, keepdims=True) \
        + d_var * (-2.0 * (x_f - mean)).mean(axis=1, keepdims=True)
    d_x = d_norm_x * rstd + d_var * 2.0 * (x_f - mean) / batch + d_mean / batch
    return d_x, d_u, d_weight, d_bias


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_norm_multiply_dropout_backward(dtype):
    batch, col = 4, 512
    x = (torch.rand([batch, col], device="npu", dtype=dtype) - 0.5).requires_grad_(True)
    u = (torch.rand([batch, col], device="npu", dtype=dtype) - 0.5).requires_grad_(True)
    weight = (torch.rand([col], device="npu", dtype=dtype) - 0.5).requires_grad_(True)
    bias = (torch.rand([col], device="npu", dtype=dtype) - 0.5).requires_grad_(True)

    golden_d_x, golden_d_u, golden_d_weight, golden_d_bias = get_golden_backward(
        x.detach().cpu().numpy(), u.detach().cpu().numpy(),
        weight.detach().cpu().numpy(), bias.detach().cpu().numpy())

    output, _, _ = ops_rec.ascendc.activation.norm_multiply_dropout(
        x, u, weight, bias, eps=EPS, dropout_ratio=0.0)
    output.sum().backward()

    assert x.grad is not None and x.grad.shape == (batch, col)
    np.testing.assert_allclose(x.grad.float().cpu().numpy(), golden_d_x, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(u.grad.float().cpu().numpy(), golden_d_u, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(weight.grad.float().cpu().numpy().reshape(1, -1), golden_d_weight,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(bias.grad.float().cpu().numpy().reshape(1, -1), golden_d_bias,
                               rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
