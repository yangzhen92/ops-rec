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
import torch_npu
import numpy as np

import ops_rec

torch.npu.config.allow_internal_format = False

# 测试有 144 种 shape 组合，增大 cache 避免 dynamic=True 时因重编译超限报错
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def get_golden(x, u, gamma, beta, eps=1e-5):
    x_f = x.astype(np.float32)
    u_f = u.astype(np.float32)
    gamma_f = gamma.astype(np.float32)
    beta_f = beta.astype(np.float32)

    mean = x_f.mean(axis=1, keepdims=True)
    var = x_f.var(axis=1, keepdims=True, ddof=0)
    rstd = 1.0 / np.sqrt(var + eps)

    y = (x_f - mean) * rstd * gamma_f + beta_f
    y = y * u_f
    return y.astype(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("a", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("r", [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_ln_mul_eager(dtype, a, r):
    torch.npu.set_device(0)
    x = torch.randn(a, r, device="npu", dtype=dtype)
    u = torch.randn(a, r, device="npu", dtype=dtype)
    gamma = torch.randn(r, device="npu", dtype=dtype)
    beta = torch.randn(r, device="npu", dtype=dtype)

    y_golden = get_golden(x.cpu().numpy(), u.cpu().numpy(), gamma.cpu().numpy(), beta.cpu().numpy())
    y_op = ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)

    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    atol = 1e-4 if dtype == torch.float32 else 1e-3
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


@pytest.mark.graph
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("a", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("r", [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_ln_mul_graph(dtype, a, r):
    torch.npu.set_device(0)
    x = torch.randn(a, r, device="npu", dtype=dtype)
    u = torch.randn(a, r, device="npu", dtype=dtype)
    gamma = torch.randn(r, device="npu", dtype=dtype)
    beta = torch.randn(r, device="npu", dtype=dtype)

    y_golden = get_golden(x.cpu().numpy(), u.cpu().numpy(), gamma.cpu().numpy(), beta.cpu().numpy())
    compiled_fn = torch_compiled(ops_rec.ascendc.activation.ln_mul, dynamic=True, fullgraph=True)
    y_op = compiled_fn(x=x, u=u, gamma=gamma, beta=beta)

    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    atol = 1e-4 if dtype == torch.float32 else 1e-3
    np.testing.assert_allclose(y_op.cpu().numpy(), y_golden, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
