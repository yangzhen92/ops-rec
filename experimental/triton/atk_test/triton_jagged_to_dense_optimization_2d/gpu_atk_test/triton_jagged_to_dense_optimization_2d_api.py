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
# pylint: disable=attribute-defined-outside-init
"""
triton_jagged_to_dense_optimization_2d 算子的ATK自定义API执行文件
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _import_local_kernel():
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from triton_jagged_to_dense_optimization_2d import (
        triton_jagged_to_dense_optimization_2d as triton_jagged_to_dense_optimization_2d_kernel,
    )

    return triton_jagged_to_dense_optimization_2d_kernel


triton_jagged_to_dense_optimization_2d = _import_local_kernel()


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _generate_jagged_offset_by_shape(shape: list, total_elements: int, device: str) -> torch.Tensor:
    """根据已知的 shape 生成确定性递增的 jagged_offset"""
    expected_length = shape[0]
    outer_size = expected_length - 1

    if outer_size <= 0:
        raise ValueError(f"Invalid shape: {shape}, outer_size must be positive")
    if total_elements <= 0:
        raise ValueError(f"Invalid total_elements: {total_elements}, must be positive")

    # 确定性算法生成递增的 offset，交替 +0/+1 增加多样性
    offsets = [0]
    current = 0
    step = max(1, total_elements // outer_size) if outer_size > 0 else total_elements

    for i in range(outer_size):
        seg_len = step + (i % 2)
        seg_len = max(1, seg_len)
        if current + seg_len > total_elements:
            seg_len = total_elements - current
        current += seg_len
        offsets.append(current)
        if current >= total_elements:
            break

    # 补齐到 expected_length 并确保最后一个值是 total_elements
    while len(offsets) < expected_length:
        offsets.append(total_elements)

    offsets[-1] = total_elements

    result = torch.tensor(offsets, dtype=torch.int64, device=device)

    # 校验生成结果
    assert result.shape[0] == expected_length, (
        f"Generated offset has wrong length: expected {expected_length}, got {result.shape[0]}"
    )
    assert result[0].item() == 0, f"offset[0] must be 0, got {result[0].item()}"
    assert result[-1].item() == total_elements, (
        f"offset[-1] must equal total_elements={total_elements}, got {result[-1].item()}"
    )
    for i in range(len(offsets) - 1):
        assert result[i].item() <= result[i + 1].item(), (
            f"offset must be non-decreasing: offset[{i}]={result[i].item()} > offset[{i + 1}]={result[i + 1].item()}"
        )

    return result


@register("triton_jagged_to_dense_optimization_2d")
class TritonJaggedToDenseOptimization2dApi(BaseApi):
    """triton_jagged_to_dense_optimization_2d 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """初始化，对齐原生代码的输入处理与 kernel 参数计算"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 获取输入
        jagged_values_input = input_data.kwargs.get("jagged_values", None)
        jagged_offsets_input = input_data.kwargs.get("jagged_offsets", None)
        jagged_max_lengths_input = input_data.kwargs.get("jagged_max_lengths", [8])
        padding_value_input = input_data.kwargs.get("padding_value", 0.0)
        operation_function_input = input_data.kwargs.get("operation_function", None)
        operation_value_input = input_data.kwargs.get("operation_value", None)

        if jagged_values_input is None:
            raise ValueError("jagged_values not found in input_data.kwargs")
        if jagged_offsets_input is None:
            raise ValueError("jagged_offsets not found in input_data.kwargs")

        # 转换到目标设备
        if not isinstance(jagged_values_input, torch.Tensor):
            jagged_values = torch.tensor(jagged_values_input, dtype=torch.float32, device=device)
        else:
            jagged_values = jagged_values_input.to(device)

        # padding_value 必须与 jagged_values 的 dtype 一致，否则 Triton 断言失败
        padding_value = torch.tensor(padding_value_input, dtype=jagged_values.dtype, device=device).item()

        # 验证维度
        if jagged_values.ndim != 2:
            raise ValueError(f"jagged_values must be 2D, got {jagged_values.ndim}D with shape {jagged_values.shape}")

        # 对齐原生代码的计算顺序：total_elements / inner_size
        total_elements = jagged_values.shape[0]
        inner_size = jagged_values.shape[1]

        # 生成 jagged_offsets（原生代码是传入的）
        if isinstance(jagged_offsets_input, torch.Tensor):
            offset_shape = list(jagged_offsets_input.shape)
        elif hasattr(jagged_offsets_input, 'shape'):
            offset_shape = list(jagged_offsets_input.shape)
        else:
            # 默认生成合理的 offset shape
            offset_shape = [min(5, total_elements + 1)]

        outer_size = offset_shape[0] - 1
        jagged_offsets = [_generate_jagged_offset_by_shape(offset_shape, total_elements, device)]

        # 处理 jagged_max_lengths（2D 情况为单元素列表）
        if isinstance(jagged_max_lengths_input, list):
            max_lengths = jagged_max_lengths_input
        elif hasattr(jagged_max_lengths_input, 'range_values'):
            rv = jagged_max_lengths_input.range_values
            if isinstance(rv, list) and len(rv) > 0:
                max_lengths = [rv[0]]  # 确定性选择第一个值
            else:
                max_lengths = [8]
        else:
            max_lengths = [8]

        if not isinstance(max_lengths, list) or len(max_lengths) != 1:
            max_lengths = [max_lengths[0] if isinstance(max_lengths, list) else max_lengths]

        # 创建 output_dense（对齐原生代码）
        output_shape = (outer_size,) + tuple(max_lengths) + (inner_size,)
        output_dense = torch.empty(output_shape, device=device, dtype=jagged_values.dtype)

        # 处理 operation_function / operation_dense
        operation_function = None
        operation_dense = None
        op_func_str = str(operation_function_input) if operation_function_input is not None else ""
        if op_func_str.lower() not in ['null', 'none', ''] and operation_value_input is not None:
            op_value = float(operation_value_input)
            operation_dense = torch.full(output_shape, op_value, dtype=jagged_values.dtype, device=device)
            operation_function = op_func_str

        # 确保 tensor 连续性
        if not jagged_values.is_contiguous():
            jagged_values = jagged_values.contiguous()
        if not jagged_offsets[0].is_contiguous():
            jagged_offsets[0] = jagged_offsets[0].contiguous()

        # 计算 kernel 参数
        thread_block_row_size = 32
        thread_block_col_size = 32
        grid = (output_dense.size(0),)  # 对应 outer_size

        dense_row_stride = output_dense.stride(-2)
        dense_matrix_stride = output_dense.stride(-3) if output_dense.ndim >= 3 else 0

        # 保存实例变量
        self._device = device
        self._jagged_values = jagged_values
        self._jagged_offsets = jagged_offsets
        self._output_dense = output_dense
        self._padding_value = padding_value
        self._operation_function = operation_function
        self._operation_dense = operation_dense
        self._grid = grid
        self._thread_block_row_size = thread_block_row_size
        self._thread_block_col_size = thread_block_col_size
        self._dense_row_stride = dense_row_stride
        self._dense_matrix_stride = dense_matrix_stride

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子，直接调用 Triton kernel"""
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 直接调用 kernel；operation_function 为 tl.constexpr，传 None 禁用 fusion
        triton_jagged_to_dense_optimization_2d[self._grid](
            self._jagged_values,
            self._jagged_offsets[0],
            self._jagged_values.stride(0),
            self._output_dense,
            self._dense_row_stride,
            self._dense_matrix_stride,
            self._thread_block_row_size,
            self._thread_block_col_size,
            padded_value=self._padding_value,
            operation_function=None,
            operation_dense=None,
        )

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output_dense
