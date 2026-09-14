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
triton_jagged_jagged_elementwise_arithmetic_ops 算子的ATK自定义API执行文件 (Autotune版本)
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


# ====== 改动1：导入路径指向 autotune/ 目录 ======
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from triton_jagged_tensor_ops import jagged_jagged_elementwise_arithmetic_ops
import triton


@register("triton_jagged_jagged_elementwise_arithmetic_ops")
class TritonJaggedJaggedElementwiseArithmeticOpsApi(BaseApi):
    """triton_jagged_jagged_elementwise_arithmetic_ops 算子的 API 封装 (Autotune版本)"""

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作

        Autotune Kernel 签名:
        def jagged_jagged_elementwise_arithmetic_ops(
            x_ptr,
            y_ptr,
            M: tl.constexpr,
            N: tl.constexpr,
            stride_row: tl.constexpr,
            stride_col: tl.constexpr,
            output,
            BLOCK_SIZE: tl.constexpr,  # autotune 参数
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
            raise ValueError("x_values and y_values must have same shape")

        # ====== 计算参数 ======
        M, N = x_values.shape
        stride_row = x_values.stride(0)
        stride_col = x_values.stride(1)

        # 创建输出 tensor
        output = torch.empty_like(x_values, device=device)

        # ====== 计算 grid（2D tile，与原生wrapper一致）======
        # thread_block_row/col_size 是autotune参数，grid从META动态读取
        # grid = cdiv(M, row) * cdiv(N, col)（与原生wrapper第98-101行一致）
        _M, _N = M, N

        def _grid(META):
            return (triton.cdiv(_M, META["thread_block_row_size"]) * triton.cdiv(_N, META["thread_block_col_size"]),)

        # ====== 保存到实例变量 ======
        self._device = device
        self._x_values = x_values
        self._y_values = y_values
        self._M = M
        self._N = N
        self._stride_row = stride_row
        self._stride_col = stride_col
        self._output = output
        self._grid = _grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子，调用 Autotune Triton kernel"""
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 调用 Autotune Triton kernel（对照原生kernel签名）======
        # thread_block_row/col_size 不传，由 @triton.autotune 自动注入
        # ops_func 必须传（非autotune参数，是add/mul功能选择）
        jagged_jagged_elementwise_arithmetic_ops[self._grid](
            self._x_values,
            self._y_values,
            self._M,
            self._N,
            self._stride_row,
            self._stride_col,
            self._output,
            # thread_block_row_size / thread_block_col_size 不传，autotune管理
            ops_func="add",  # 非autotune参数，必须传
        )

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output
