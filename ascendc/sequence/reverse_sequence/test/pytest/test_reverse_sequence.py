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
import numpy as np
import torch_npu
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

BATCH_SIZE_DIM = [40, 78, 128]  # bs维度
MAX_SEQ_LEN_DIM = [123, 500, 4000]  # max_seq_len维度
DATA_DIM = [1, 2, 16, 32]  # data_dim 维度, 包含16的倍数和非16的倍数
DIM_LIST = list(itertools.product(BATCH_SIZE_DIM, MAX_SEQ_LEN_DIM, DATA_DIM))

_PRECISION_ERROR_RANGE = {
    torch.float32: 1e-4,
    torch.float16: 1e-3,
    torch.bfloat16: 5e-3,
    torch.int64: 1e-4,
}
INPUT_DATA_TYPE = _PRECISION_ERROR_RANGE.keys()
SEQ_LEN_DATA_TYPE = [torch.int64, torch.int32]
TYPE_LIST = list(itertools.product(INPUT_DATA_TYPE, SEQ_LEN_DATA_TYPE))

# 边界测试用例
EDGE_CASE_DIMS = [
    (1, 1, 16),  # 最小batch_size, max_seq_len, 特征维度
    (10240, 1, 16),  # 最大的bs
    (1, 102400, 16),  # 最大的 max_seq_len
    (51, 23, 1024),  # 最大的 data_dim
    (100, 10000, 512),  # 较大的batch和特征维度
    (123, 234, 3),  # 非对齐的data_dim维度
]

# 边界测试用例 - 超出范围
OUT_OF_RANGE_CASE_DIMS = [
    (10241, 1, 16),  # bs维度超出范围
    (5, 102401, 16),  # max_seq_len维度超出范围
    (10, 10, 1025),  # data_dim维度超出范围
]

np.random.seed(42)


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def reverse_sequence_python(input_data, seq_lengths):
    """
    input_data: [bs, max_seq_len, data_dim]  输入序列
    seq_lengths: [bs]        每行实际长度
    return: [bs, max_seq_len, data_dim]  翻转后序列

    向量化实现（gather）：避免逐 batch 循环中 flip/切片赋值/.item() 同步
    在 NPU 上阻塞的问题，同时大幅缩短测试时间。
    """
    bs, max_seq_len, data_dim = input_data.shape
    positions = torch.arange(max_seq_len, device=input_data.device).unsqueeze(0)  # [1, max_seq_len]
    len_b = seq_lengths.unsqueeze(1)  # [bs, 1]
    # 行号 < seq_len 时映射到逆序位置 len_b - 1 - i，否则保持原行号
    index = torch.where(positions < len_b, len_b - 1 - positions, positions)  # [bs, max_seq_len]
    index = index.unsqueeze(-1).expand(-1, -1, data_dim).to(torch.int64)
    return torch.gather(input_data, 1, index)


class ReverseSequenceTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, seq_lengths):
        ctx.save_for_backward(seq_lengths)
        return reverse_sequence_python(input_data, seq_lengths)

    @staticmethod
    def backward(ctx, grad_input):
        saved_data = list(ctx.saved_tensors)
        seq_lengths = saved_data[0]
        grad_output = reverse_sequence_python(grad_input, seq_lengths)
        return grad_output, None


def reverse_sequence_torch(input_data, seq_lengths):
    return ReverseSequenceTorch.apply(input_data, seq_lengths)


def reverse_sequence_npu(input_data, seq_lengths):
    return ops_rec.ascendc.sequence.reverse_sequence(input_data, seq_lengths)


def get_result(device, input_data, seq_lengths, types):
    """获取指定设备上的算子执行结果"""
    input_dt, seq_lengths_dt = types
    input_data = torch.tensor(input_data, dtype=input_dt, device=device)
    seq_lengths = torch.tensor(seq_lengths, dtype=seq_lengths_dt, device=device)
    if device.type == "cpu":
        output_data = reverse_sequence_torch(input_data, seq_lengths)
    else:
        output_data = reverse_sequence_npu(input_data, seq_lengths)
        output_data = output_data.cpu()
    return output_data


def compare_results(golden_result, npu_result, tolerance):
    assert golden_result.shape == npu_result.shape, \
        f"Shape mismatch: golden {golden_result.shape} vs npu {npu_result.shape}"

    if golden_result.numel() > 0:
        result_forward = torch.abs(golden_result - npu_result) < tolerance
        assert result_forward.all().item(), "Result values do not match within tolerance"
    else:
        assert torch.equal(golden_result, npu_result), "Empty tensors should be equal"


def generate_test_data(dim0, dim1, dim2):
    """生成测试数据"""
    input_data = np.random.randn(dim0, dim1, dim2).astype(np.float32)
    # 确保 len(seq_lengths) == input_data.shape[0]
    # 生成的length值的范围需要包含最大值
    seq_lengths = np.random.randint(0, dim1 + 1, dim0)
    return input_data, seq_lengths


def run_test_and_compare(input_data, seq_lengths, types):
    """运行测试的核心逻辑"""
    # 获取结果
    npu_result = get_result(torch.device(DEVICE), input_data, seq_lengths, types)
    golden_result = get_result(torch.device("cpu"), input_data, seq_lengths, types)

    # 根据数据类型获取相应的容差值
    tolerance = _PRECISION_ERROR_RANGE[types[0]]
    compare_results(golden_result, npu_result, tolerance)


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
def test_reverse_sequence(dims, types):
    """基本功能测试（前向 + 反向）"""
    logging.info("=== case info: dims: %s, types:%s ===", dims, types)
    # 生成数据，算子调用，比较结果
    bs, max_seq_len, data_dim = dims
    input_dt, sql_len_dt = types
    input_data_npu = torch.rand(dims, dtype=input_dt, device=DEVICE)
    input_data_npu_clone = input_data_npu.clone()
    seq_lengths = torch.randint(0, max_seq_len + 1, (bs,), dtype=sql_len_dt, device=torch.device(DEVICE))
    is_int64 = types[0] == torch.int64
    # 融合算子实现 前向
    if not is_int64:
        input_data_npu.requires_grad_(True)
        input_data_npu_clone.requires_grad_(True)
    fused_op_ret = reverse_sequence_npu(input_data_npu, seq_lengths)
    if not is_int64:
        npu_op_loss = torch.sum(fused_op_ret)
        npu_op_loss.backward()
        npu_op_grad = input_data_npu.grad.cpu()
        input_data_npu.grad = None
    # pytorch 实现 前向
    pytorch_ret = reverse_sequence_torch(input_data_npu_clone, seq_lengths)
    if not is_int64:
        npu_torch_loss = torch.sum(pytorch_ret)
        npu_torch_loss.backward()
        npu_torch_grad = input_data_npu_clone.grad.cpu()
        input_data_npu.grad = None
    # 比较前向结果（搬到 CPU 比较，避免比对算子在 NPU 同步点阻塞）
    compare_results(fused_op_ret.cpu(), pytorch_ret.cpu(), _PRECISION_ERROR_RANGE[types[0]])

    if is_int64:
        # torch.int64只支持前向
        return
    # 比较融合算子和pytorch实现的反向
    assert torch.allclose(
        npu_op_grad,
        npu_torch_grad,
        atol=_PRECISION_ERROR_RANGE[types[0]],
        rtol=_PRECISION_ERROR_RANGE[types[0]]
    ), f"golden and npu fused op result is not closed. CASE INFO - dims:{dims}, types:{types}"


@pytest.mark.graph
@pytest.mark.parametrize("dims", [(40, 123, 16), (78, 500, 2), (128, 4000, 32)])
def test_reverse_sequence_graph(dims):
    """torch.compile 图模式测试"""
    logging.info("=== graph case info: dims: %s ===", dims)
    bs, max_seq_len, data_dim = dims
    input_data_npu = torch.rand(dims, dtype=torch.float32, device=DEVICE)
    seq_lengths = torch.randint(0, max_seq_len + 1, (bs,), dtype=torch.int64, device=torch.device(DEVICE))

    compiled_fn = torch_compiled(ops_rec.ascendc.sequence.reverse_sequence, dynamic=True, fullgraph=True)
    graph_ret = compiled_fn(input_data_npu, seq_lengths)
    pytorch_ret = reverse_sequence_python(input_data_npu, seq_lengths)

    # 先搬到 CPU 再比较：编译图输出与 eager 输出直接在 NPU 上比对会在同步点阻塞
    compare_results(graph_ret.cpu(), pytorch_ret.cpu(), _PRECISION_ERROR_RANGE[torch.float32])


@pytest.mark.parametrize("dims", EDGE_CASE_DIMS)
def test_valid_edge_case(dims):
    bs, max_seq_len, data_dim = dims
    input_data, seq_lengths = generate_test_data(bs, max_seq_len, data_dim)
    run_test_and_compare(input_data, seq_lengths, (torch.float32, torch.int64))


@pytest.mark.parametrize("dims", OUT_OF_RANGE_CASE_DIMS)
def test_invalid_range_cases(dims):
    bs, max_seq_len, data_dim = dims
    input_data, seq_lengths = generate_test_data(bs, max_seq_len, data_dim)
    with pytest.raises(RuntimeError):
        run_test_and_compare(input_data, seq_lengths, (torch.float32, torch.int64))


@pytest.mark.parametrize("dims", [[10, 10, 16]])
@pytest.mark.parametrize("input_dtype", [torch.int8, torch.int32])
@pytest.mark.parametrize("seq_len_dtype", [torch.int8])
def test_invalid_data_type_cases(dims, input_dtype, seq_len_dtype):
    bs, max_seq_len, data_dim = dims
    input_data, seq_lengths = generate_test_data(bs, max_seq_len, data_dim)
    types = (input_dtype, seq_len_dtype)
    with pytest.raises(RuntimeError):
        run_test_and_compare(input_data, seq_lengths, types)


@pytest.mark.ci
def test_reverse_sequence_ci():
    """CI 快速冒烟用例：固定 shape + float32，仅验证正确性"""
    torch.npu.set_device(DEVICE)
    dims = (10, 16, 16)
    bs, max_seq_len, _ = dims
    input_data_npu = torch.rand(dims, dtype=torch.float32, device=DEVICE)
    seq_lengths = torch.randint(0, max_seq_len + 1, (bs,), dtype=torch.int64, device=torch.device(DEVICE))

    npu_ret = reverse_sequence_npu(input_data_npu, seq_lengths)
    pytorch_ret = reverse_sequence_python(input_data_npu, seq_lengths)

    compare_results(npu_ret.cpu(), pytorch_ret.cpu(), _PRECISION_ERROR_RANGE[torch.float32])


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    pytest.main([__file__, "-sv"])
