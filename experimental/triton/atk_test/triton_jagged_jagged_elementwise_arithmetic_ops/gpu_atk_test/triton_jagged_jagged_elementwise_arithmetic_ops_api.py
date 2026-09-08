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

"""
triton_jagged_jagged_elementwise_arithmetic_ops 算子的ATK自定义API执行文件
"""
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def get_device():
    """获取可用的设备，优先NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'


# 从被测代码直接引用 Triton kernel
# 注意：需要确保 triton_jagged_tensor_ops.py 在当前目录或 Python 路径中
from triton_jagged_tensor_ops import jagged_jagged_elementwise_arithmetic_ops
import triton


@register("triton_jagged_jagged_elementwise_arithmetic_ops")
class TritonJaggedJaggedElementwiseArithmeticOpsApi(BaseApi):
    """triton_jagged_jagged_elementwise_arithmetic_ops 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作

        Kernel 签名:
        def jagged_jagged_elementwise_arithmetic_ops(
            x_ptr,           # x_values 指针
            y_ptr,           # y_values 指针
            M: tl.constexpr, # 行数
            N: tl.constexpr, # 列数
            stride_row: tl.constexpr,  # 行 stride
            stride_col: tl.constexpr,  # 列 stride
            output,          # 输出指针
            thread_block_row_size: tl.constexpr,  # thread block 行大小
            thread_block_col_size: tl.constexpr,  # thread block 列大小
            ops_func: tl.constexpr,  # 运算类型 "add" 或 "mul"
        )
        """
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 获取输入 ======
        x_values_input = input_data.kwargs.get("x_values", None)
        y_values_input = input_data.kwargs.get("y_values", None)

        if x_values_input is None:
            raise ValueError("x_values not found in input_data.kwargs")
        if y_values_input is None:
            raise ValueError("y_values not found in input_data.kwargs")

        # ====== 转换到目标设备 ======
        if not isinstance(x_values_input, torch.Tensor):
            x_values = torch.tensor(x_values_input, dtype=torch.float32, device=device)
        else:
            x_values = x_values_input.to(device)

        if not isinstance(y_values_input, torch.Tensor):
            y_values = torch.tensor(y_values_input, dtype=torch.float32, device=device)
        else:
            y_values = y_values_input.to(device)

        # ====== 验证维度 ======
        if x_values.ndim != 2:
            raise ValueError(f"x_values must be 2D, got {x_values.ndim}D")
        if y_values.ndim != 2:
            raise ValueError(f"y_values must be 2D, got {y_values.ndim}D")
        if x_values.shape != y_values.shape:
            raise ValueError(f"x_values and y_values must have same shape, got {x_values.shape} vs {y_values.shape}")

        # ====== 计算参数 ======
        M, N = x_values.shape
        stride_row = x_values.stride(0)
        stride_col = x_values.stride(1)

        # thread block 参数（与原生代码一致）
        thread_block_row_size = 32
        thread_block_col_size = 32
        ops_func = "add"

        # 创建输出 tensor
        output = torch.empty_like(x_values, device=device)

        # ====== 计算 grid（与原生代码一致）======
        # grid = cdiv(M, row_size) * cdiv(N, col_size)
        grid = (
            triton.cdiv(M, thread_block_row_size)
            * triton.cdiv(N, thread_block_col_size),
        )

        # ====== 保存到实例变量 ======
        self._device = device
        self._x_values = x_values
        self._y_values = y_values
        self._M = M
        self._N = N
        self._stride_row = stride_row
        self._stride_col = stride_col
        self._output = output
        self._thread_block_row_size = thread_block_row_size
        self._thread_block_col_size = thread_block_col_size
        self._ops_func = ops_func
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子，直接调用 Triton kernel"""
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 直接调用 Triton kernel ======
        jagged_jagged_elementwise_arithmetic_ops[self._grid](
            self._x_values,
            self._y_values,
            self._M,
            self._N,
            self._stride_row,
            self._stride_col,
            self._output,
            self._thread_block_row_size,
            self._thread_block_col_size,
            ops_func=self._ops_func,
        )

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output