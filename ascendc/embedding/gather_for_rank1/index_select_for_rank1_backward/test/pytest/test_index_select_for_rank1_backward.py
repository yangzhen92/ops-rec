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
import itertools
import logging
import sys

import pytest
import torch

import ops_rec

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)

torch.npu.config.allow_internal_format = False

# 模块导入时即完成设备初始化：避免首个用例内惰性初始化与运行时子系统
# （执行器缓存/流/OPP 加载）的初始化顺序问题导致间歇性 kernel 异常
torch.npu.set_device(DEVICE)

# 测试 shape 组合较多，增大 cache 避免 dynamic=True 时因重编译超限报错
torch._dynamo.config.cache_size_limit = 256

# 权重表长度（grad_x 长度）：包含最小值/非对齐/较大值；
# 注意 kernel 需将 grad_x 全量缓存进 UB，过大的 x_dim0 会超出 UB 容量
X_DIM0_LIST = [1, 97, 1024, 16384]
# index/grad_y 长度
INDEX_LEN_LIST = [1, 100, 4096]
DIM_LIST = list(itertools.product(X_DIM0_LIST, INDEX_LEN_LIST))

# grad_y/x 仅支持 float32；index 支持 int32/int64
GRAD_DTYPE = torch.float32
INDEX_DTYPE_LIST = [torch.int32, torch.int64]
TYPE_LIST = list(itertools.product([GRAD_DTYPE], INDEX_DTYPE_LIST))

_PRECISION_ERROR_RANGE = 1e-4


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def index_select_for_rank1_backward_npu(grad_y, x, index):
    """NPU 融合算子路径：返回 (grad_x, grad_index)，grad_index 恒为 undefined"""
    return ops_rec.ascendc.embedding.index_select_for_rank1_backward(grad_y, x, index)


def index_select_for_rank1_backward_golden(grad_y, x, index):
    """CPU 参考实现：grad_x 清零后按 index scatter-add 累加 grad_y"""
    grad_x = torch.zeros_like(x, dtype=torch.float32)
    grad_x.scatter_add_(0, index.to(torch.int64), grad_y)
    return grad_x


def compare_results(golden_result, npu_result, tolerance):
    assert golden_result.shape == npu_result.shape, (
        f"Shape mismatch: golden {golden_result.shape} vs npu {npu_result.shape}"
    )

    if golden_result.numel() > 0:
        result = torch.abs(golden_result - npu_result) < tolerance
        assert result.all().item(), "Result values do not match within tolerance"
    else:
        assert torch.equal(golden_result, npu_result), "Empty tensors should be equal"


def generate_test_data(x_dim0, idx_len, index_dtype):
    """生成测试数据（CPU 侧生成后按需搬运到 NPU）"""
    grad_y = torch.randn(idx_len, dtype=GRAD_DTYPE)
    x = torch.randn(x_dim0, dtype=GRAD_DTYPE)
    index = torch.randint(0, x_dim0, (idx_len,), dtype=index_dtype)
    return grad_y, x, index


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
def test_index_select_for_rank1_backward(dims, types):
    """基本功能测试：NPU 融合算子 vs CPU golden（scatter_add）"""
    logging.info("=== case info: dims: %s, types: %s ===", dims, types)
    x_dim0, idx_len = dims
    _, index_dtype = types
    grad_y_cpu, x_cpu, index_cpu = generate_test_data(x_dim0, idx_len, index_dtype)

    grad_y_npu = grad_y_cpu.to(DEVICE)
    x_npu = x_cpu.to(DEVICE)
    index_npu = index_cpu.to(DEVICE)

    grad_x, grad_index = index_select_for_rank1_backward_npu(grad_y_npu, x_npu, index_npu)
    # 第二个返回值为 index 梯度占位，恒为 undefined
    assert grad_index is None, "grad_index should be undefined (None)"

    golden = index_select_for_rank1_backward_golden(grad_y_cpu, x_cpu, index_cpu)
    # 搬到 CPU 比较，避免比对算子在 NPU 同步点阻塞
    compare_results(golden, grad_x.cpu(), _PRECISION_ERROR_RANGE)


@pytest.mark.graph
@pytest.mark.parametrize("dims", [(97, 100), (1024, 4096)])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPE_LIST)
def test_index_select_for_rank1_backward_graph(dims, index_dtype):
    """torch.compile 图模式测试（Meta 实现提供 shape 推断）"""
    logging.info("=== graph case info: dims: %s, index_dtype: %s ===", dims, index_dtype)
    x_dim0, idx_len = dims
    grad_y_cpu, x_cpu, index_cpu = generate_test_data(x_dim0, idx_len, index_dtype)

    def fn(grad_y, x, index):
        grad_x, _ = torch.ops.ops_rec.index_select_for_rank1_backward(grad_y, x, index)
        return grad_x

    compiled_fn = torch_compiled(fn, dynamic=True, fullgraph=True)
    graph_ret = compiled_fn(grad_y_cpu.to(DEVICE), x_cpu.to(DEVICE), index_cpu.to(DEVICE))

    golden = index_select_for_rank1_backward_golden(grad_y_cpu, x_cpu, index_cpu)
    compare_results(golden, graph_ret.cpu(), _PRECISION_ERROR_RANGE)


@pytest.mark.parametrize("dims", [(100, 100)])
def test_invalid_dim_cases(dims):
    """非 1D 输入：torch 扩展层维度校验应拒绝"""
    x_dim0, idx_len = dims
    grad_y, x, index = generate_test_data(x_dim0, idx_len, torch.int32)

    # x 为 2D
    x_2d = x.reshape(10, 10)
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y.to(DEVICE), x_2d.to(DEVICE), index.to(DEVICE))

    # grad_y 为 2D
    grad_y_2d = grad_y.reshape(10, 10)
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y_2d.to(DEVICE), x.to(DEVICE), index.to(DEVICE))

    # index 为 2D
    index_2d = index.reshape(10, 10)
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y.to(DEVICE), x.to(DEVICE), index_2d.to(DEVICE))


@pytest.mark.parametrize("dims", [(100, 100)])
def test_invalid_grad_index_mismatch(dims):
    """grad_y 与 index 长度不一致：应拒绝"""
    x_dim0, idx_len = dims
    grad_y, x, index = generate_test_data(x_dim0, idx_len, torch.int32)
    grad_y_short = grad_y[:-1]
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y_short.to(DEVICE), x.to(DEVICE), index.to(DEVICE))


@pytest.mark.parametrize("dims", [(100, 100)])
@pytest.mark.parametrize("grad_dtype", [torch.float16, torch.bfloat16])
def test_invalid_grad_dtype_cases(dims, grad_dtype):
    """grad_y/x 不支持的数据类型：aclnn 校验应拒绝"""
    x_dim0, idx_len = dims
    grad_y, x, index = generate_test_data(x_dim0, idx_len, torch.int32)
    grad_y_bad = grad_y.to(grad_dtype)
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y_bad.to(DEVICE), x.to(DEVICE), index.to(DEVICE))

    x_bad = x.to(grad_dtype)
    with pytest.raises(RuntimeError):
        index_select_for_rank1_backward_npu(grad_y.to(DEVICE), x_bad.to(DEVICE), index.to(DEVICE))


@pytest.mark.ci
def test_index_select_for_rank1_backward_ci():
    """CI 快速冒烟用例：固定 shape + float32/int32，仅验证正确性"""
    torch.npu.set_device(DEVICE)
    grad_y, x, index = generate_test_data(1024, 4096, torch.int32)

    grad_x, _ = index_select_for_rank1_backward_npu(grad_y.to(DEVICE), x.to(DEVICE), index.to(DEVICE))
    golden = index_select_for_rank1_backward_golden(grad_y, x, index)
    compare_results(golden, grad_x.cpu(), _PRECISION_ERROR_RANGE)


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    pytest.main([__file__, "-sv"])
