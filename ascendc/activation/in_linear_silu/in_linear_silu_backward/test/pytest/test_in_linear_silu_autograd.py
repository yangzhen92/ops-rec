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
"""in_linear_silu 算子 autograd 测试（eager + inductor）。

测试链路:
    forward: distance_in_linear_silu(x, weight, bias, split_args)
        -> (user, value, query, key, linear_output)
    loss = MSE((user + value + query + key) / 4, target)
    loss.backward()
    对比 x_grad / weight_grad / bias_grad 与 CPU golden

autograd 注册通过 torch.library.register_autograd 完成（见 torch_extension/_autograd.py）。
inductor 模式用例使用 torch.compile(backend="inductor") 验证图模式下的 autograd 正确性。

注意: 前向算子 distance_in_linear_silu 的 bias 为 REQUIRED（非可选），数据类型固定为
float32，因此本测试始终传入 float32 的 bias，不测试 enable_bias=False 场景。
enable_bias=False 的反向算子测试见 test_in_linear_silu_backward.py。
"""

import importlib.util
import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ops_rec

torch.npu.config.allow_internal_format = False
torch._dynamo.config.cache_size_limit = 256

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# autograd 注册：从 torch_extension/_autograd.py 导入并注册
# ---------------------------------------------------------------------------
def _load_autograd_module():
    """通过文件路径导入 torch_extension/_autograd.py 并注册 autograd。"""
    # 测试文件: in_linear_silu_backward/test/pytest/test_in_linear_silu_autograd.py
    # 目标文件: torch_extension/_autograd.py
    autograd_path = Path(__file__).resolve().parents[3] / "torch_extension" / "_autograd.py"
    if not autograd_path.is_file():
        raise FileNotFoundError(f"_autograd.py not found at {autograd_path}")
    spec = importlib.util.spec_from_file_location("_in_linear_silu_autograd", autograd_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.register_autograd()


_load_autograd_module()


def set_seed(seed=0):
    """设置随机种子以确保可重复性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.npu.manual_seed_all(seed)


def allclose(tensor, other, atol, ratio, name):
    """逐元素对比，允许 ratio 比例的元素超出 atol。"""
    assert tensor.shape == other.shape, f"{name} shape mismatch: {tensor.shape} vs {other.shape}"
    close = torch.isclose(tensor.to(torch.float32), other.to(torch.float32), rtol=atol, atol=atol)
    diff_count = tensor.numel() - torch.sum(close)
    is_pass = (diff_count / tensor.numel()) <= ratio
    if not is_pass:
        logger.warning("%s: %d/%d elements exceed atol=%s", name, diff_count, tensor.numel(), atol)
    return is_pass


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------
def get_data_autograd(total_seqs, embeddim_dim, hidden_size, data_type, is_diff_dim):
    h = hidden_size // 4
    if is_diff_dim:
        split_args = [h - 32, h - 32, h + 32, h + 32]
    else:
        split_args = [h] * 4
    N = sum(split_args)

    x = torch.rand(total_seqs, embeddim_dim, dtype=data_type)
    weight = torch.rand(N, embeddim_dim, dtype=data_type)
    target = torch.rand(total_seqs, split_args[0], dtype=data_type)
    # bias 数据类型固定为 float32（前向算子要求）
    bias = torch.rand(N, dtype=torch.float32)
    return x, weight, bias, target, split_args


# ---------------------------------------------------------------------------
# NPU autograd 链路
# ---------------------------------------------------------------------------
def in_linear_silu_autograd_npu(x, weight, bias, target, split_args, backend="eager"):
    """NPU 上通过 autograd 计算 x/weight/bias 的梯度。"""
    x = torch.nn.Parameter(x.clone().to("npu"), requires_grad=True)
    weight = torch.nn.Parameter(weight.clone().to("npu"), requires_grad=True)
    bias = torch.nn.Parameter(bias.clone().to("npu"), requires_grad=True)

    target_npu = target.to("npu")

    def _forward(x, weight, bias, split_args):
        return ops_rec.ascendc.activation.distance_in_linear_silu(x, weight, bias, split_args)

    if backend == "inductor":
        compiled_fn = torch.compile(_forward, backend="inductor", dynamic=True, fullgraph=False)
        user, value, query, key, _ = compiled_fn(x, weight, bias, split_args)
    else:
        user, value, query, key, _ = _forward(x, weight, bias, split_args)

    output = (user + value + query + key) / 4
    loss = nn.MSELoss()(output, target_npu)
    loss.backward()
    torch.npu.synchronize()

    x_grad = x.grad.detach().cpu().to(torch.float32)
    weight_grad = weight.grad.detach().cpu().to(torch.float32)
    bias_grad = bias.grad.detach().cpu().to(torch.float32)
    return x_grad, weight_grad, bias_grad


# ---------------------------------------------------------------------------
# CPU golden
# ---------------------------------------------------------------------------
def in_linear_silu_autograd_golden(x, weight, bias, target, split_args):
    """CPU 上的 golden 实现：F.linear + F.silu + split。"""
    x = torch.nn.Parameter(x.clone().to(torch.float32), requires_grad=True)
    weight = torch.nn.Parameter(weight.clone().to(torch.float32), requires_grad=True)
    bias = torch.nn.Parameter(bias.clone().to(torch.float32), requires_grad=True)

    mixed_uvqk = F.linear(x, weight, bias)
    mixed_uvqk = F.silu(mixed_uvqk)
    user, value, query, key = torch.split(mixed_uvqk, split_args, dim=-1)

    output = (user + value + query + key) / 4
    loss = nn.MSELoss()(output, target.to(output.dtype))
    loss.backward()

    x_grad = x.grad.detach().cpu().to(torch.float32)
    weight_grad = weight.grad.detach().cpu().to(torch.float32)
    bias_grad = bias.grad.detach().cpu().to(torch.float32)
    return x_grad, weight_grad, bias_grad


# ---------------------------------------------------------------------------
# 对比执行
# ---------------------------------------------------------------------------
def execute(total_seqs, dim, data_type, is_diff_dim, backend):
    if dim[0] <= 32 and is_diff_dim:
        return
    x, weight, bias, target, split_args = get_data_autograd(total_seqs, dim[0], dim[1], data_type, is_diff_dim)

    x_grad_golden, weight_grad_golden, bias_grad_golden = in_linear_silu_autograd_golden(
        x, weight, bias, target, split_args
    )
    x_grad, weight_grad, bias_grad = in_linear_silu_autograd_npu(x, weight, bias, target, split_args, backend)

    torch.npu.synchronize()
    atol = 1e-4
    if data_type == torch.float16:
        atol = 1e-3
    elif data_type == torch.bfloat16:
        atol = 5e-3

    ratio = 1e-3
    x_res = allclose(x_grad, x_grad_golden, atol, ratio, "x_grad")
    weight_res = allclose(weight_grad, weight_grad_golden, atol, ratio, "weight_grad")
    bias_res = allclose(bias_grad, bias_grad_golden, atol, ratio, "bias_grad")

    assert x_res and weight_res and bias_res, (
        f"autograd check failed (backend={backend}, total_seqs={total_seqs}, dim={dim}, "
        f"dtype={data_type}, is_diff_dim={is_diff_dim})"
    )


# ---------------------------------------------------------------------------
# Eager 模式测试
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_seqs", [1, 1123, 2048, 3201, 7687, 20480])
@pytest.mark.parametrize("dim", [(16, 128), (64, 512), (240, 960), (128, 2048)])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False, True])
def test_in_linear_silu_autograd_eager(total_seqs, dim, data_type, is_diff_dim):
    set_seed(0)
    execute(total_seqs, dim, data_type, is_diff_dim, "eager")


@pytest.mark.parametrize("total_seqs", [511, 2048])
@pytest.mark.parametrize("dim", [(8160, 32640), (512, 6144), (1024, 16384)])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_autograd_large(total_seqs, dim, data_type, is_diff_dim):
    set_seed(0)
    execute(total_seqs, dim, data_type, is_diff_dim, "eager")


# ---------------------------------------------------------------------------
# Inductor 模式测试（torch.compile + autograd）
# ---------------------------------------------------------------------------
@pytest.mark.graph
@pytest.mark.parametrize("total_seqs", [1, 511, 2048])
@pytest.mark.parametrize("dim", [(16, 256), (64, 512), (512, 2048)])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_autograd_inductor(total_seqs, dim, data_type, is_diff_dim):
    """inductor 图模式下 autograd 正确性验证。"""
    set_seed(0)
    execute(total_seqs, dim, data_type, is_diff_dim, "inductor")


@pytest.mark.ci
def test_in_linear_silu_autograd_ci():
    set_seed(0)
    execute(16, (16, 256), torch.float16, False, "eager")


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
