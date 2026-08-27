#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""token_mixing pytest（对齐 RecSDK cust_op/test/token_mixing/test_token_mixing.py）。

相对旧仓：
- 用例 shape / gamma=ones / beta=zeros / golden=torch.nn.LayerNorm / atol=1e-6 保持一致
  （np.allclose 未传 rtol 时使用 numpy 默认 rtol=1e-5）
- 调用改为 ops_rec.ascendc.mixing.token_mixing
- 额外增加 graph（与 eager 同参）与 ci 冒烟

"""

import sys
import numpy as np
import pytest
import torch
import ops_rec

torch.npu.config.allow_internal_format = False
torch._dynamo.config.cache_size_limit = 256

RECSDK_B = [1, 45, 256, 512, 1024]
RECSDK_S = [128, 256, 338, 507, 512, 1024]
DEFAULT_EPS = 1e-7


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def get_global_loss(x):
    """CPU golden：与 RecSDK 相同，用 torch.nn.LayerNorm(eps=1e-7)。"""
    input_tensor = torch.from_numpy(x)
    transpose_tensor = torch.from_numpy(x).permute(0, 2, 1)
    add = torch.add(input_tensor, transpose_tensor)
    layer_norm = torch.nn.LayerNorm(add.size()[2:], eps=DEFAULT_EPS)
    result = layer_norm(add)
    return result.cpu().detach().numpy()


def get_fused_loss_op(x, gamma, beta, device, compiled_fn=None):
    """NPU 调用：对齐 RecSDK get_fused_loss_op，改走 ops_rec。"""
    torch.npu.set_device(device)
    x_t = torch.from_numpy(x).to(device)
    gamma_t = torch.from_numpy(gamma).to(device)
    beta_t = torch.from_numpy(beta).to(device)
    fn = compiled_fn if compiled_fn is not None else ops_rec.ascendc.mixing.token_mixing
    result = fn(x_t, gamma_t, beta_t, DEFAULT_EPS)
    return result.cpu().detach().numpy()


def _run_recsdk_case(B, S, device="npu:0", compiled_fn=None):
    tensor = np.random.randn(B, S, S).astype(np.float32)
    gamma = np.ones(S, dtype=np.float32)
    beta = np.zeros(S, dtype=np.float32)
    result = get_global_loss(tensor)
    result1 = get_fused_loss_op(tensor, gamma, beta, device, compiled_fn=compiled_fn)
    # 与 RecSDK 完全一致：只传 atol=1e-6，rtol 用 numpy 默认 1e-5
    assert np.allclose(result, result1, atol=1e-6)


@pytest.mark.parametrize("B", RECSDK_B)
@pytest.mark.parametrize("S", RECSDK_S)
@pytest.mark.parametrize("device", ["npu:0"])
def test_token_mixing_eager(B, S, device):
    """eager：用例与精度对齐 RecSDK test_token_mixing。"""
    _run_recsdk_case(B, S, device)


@pytest.mark.graph
@pytest.mark.parametrize("B", RECSDK_B)
@pytest.mark.parametrize("S", RECSDK_S)
@pytest.mark.parametrize("device", ["npu:0"])
def test_token_mixing_graph(B, S, device):
    """graph：与 eager 完全相同的 B/S/device/精度。"""
    compiled_fn = torch_compiled(ops_rec.ascendc.mixing.token_mixing, dynamic=None, fullgraph=False, backend="inductor")
    _run_recsdk_case(B, S, device, compiled_fn=compiled_fn)


@pytest.mark.ci
def test_token_mixing_ci():
    """CI 快速冒烟：小 shape，精度仍用 RecSDK 的 allclose(atol=1e-6)。"""
    _run_recsdk_case(2, 16, "npu:0")


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
