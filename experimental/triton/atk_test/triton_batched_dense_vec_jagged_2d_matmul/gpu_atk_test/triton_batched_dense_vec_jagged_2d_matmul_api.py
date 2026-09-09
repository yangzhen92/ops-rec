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

# triton_batched_dense_vec_jagged_2d_matmul_api.py
"""
triton_batched_dense_vec_jagged_2d_matmul 算子的ATK自定义API执行文件

基于原生实现：batched_dense_vec_jagged_2d_matmul
关键约束：
  B = offset.size(0) - 1
  H = dense.size(0) // B
  D = values.size(-1) // H
"""

import triton
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_batched_dense_vec_jagged_2d_matmul import triton_batched_dense_vec_jagged_2d_matmul


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


@register("triton_batched_dense_vec_jagged_2d_matmul")
class TritonBatchedDenseVecJagged2dMatmulApi(BaseApi):
    # pylint: disable=attribute-defined-outside-init
    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 获取输入 ======
        dense_input = input_data.kwargs.get("dense", None)
        values_input = input_data.kwargs.get("values", None)
        offset_input = input_data.kwargs.get("offset", None)

        if dense_input is None:
            raise ValueError("dense not found in input_data.kwargs")
        if values_input is None:
            raise ValueError("values not found in input_data.kwargs")
        if offset_input is None:
            raise ValueError("offset not found in input_data.kwargs")

        # ====== 转换到目标设备 ======
        if not isinstance(dense_input, torch.Tensor):
            dense = torch.tensor(dense_input, dtype=torch.float32, device=device)
        else:
            dense = dense_input.to(device=device, dtype=torch.float32)

        if not isinstance(values_input, torch.Tensor):
            values = torch.tensor(values_input, dtype=torch.float32, device=device)
        else:
            values = values_input.to(device=device, dtype=torch.float32)

        if not isinstance(offset_input, torch.Tensor):
            offset_raw = torch.tensor(offset_input, dtype=torch.int64, device=device)
        else:
            offset_raw = offset_input.to(device)

        # ====== 验证维度 ======
        if dense.ndim != 2:
            raise ValueError(f"dense must be 2D [B*H, max_L], got {dense.ndim}D")
        if values.ndim != 2:
            raise ValueError(f"values must be 2D [total_L, H*D], got {values.ndim}D")
        if offset_raw.ndim != 1:
            raise ValueError(f"offset must be 1D [B+1], got {offset_raw.ndim}D")

        # 维度语义：dense=[B*H, max_L], values=[total_L, H*D], output=[B*H, D]
        # B = offset.size(0) - 1
        B = offset_raw.size(0) - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        # H = dense.size(0) // B（dense第一维是 B*H）
        H = dense.size(0) // B

        # D = values.size(-1) // H（D 来自 values 的第二维 H*D，不是 dense）
        D = values.size(-1) // H

        # 验证计算结果
        BH = B * H
        HD = H * D

        if dense.shape[0] != BH:
            raise ValueError(f"dense.shape[0] should be B*H={BH}, got {dense.shape[0]}")
    
        if values.shape[1] != HD:
            raise ValueError(f"values.shape[1] should be H*D={HD}, got {values.shape[1]}")

        total_L = values.shape[0]
        offset = self._build_valid_offset(B, total_L, device)

        output = torch.empty((BH, D), dtype=values.dtype, device=device)

        # ====== 计算 grid ======
        thread_block_col_size = 32
        GRID_DIM_COL = triton.cdiv(D, thread_block_col_size)
        grid = (BH * GRID_DIM_COL,)

        # ====== 保存到实例变量 ======
        self._device = device
        self._dense = dense
        self._values = values
        self._offset = offset
        self._B = B
        self._H = H
        self._D = D
        self._output = output
        self._grid = grid
        self._thread_block_col_size = thread_block_col_size

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行算子，直接调用 Triton kernel"""
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 直接调用 Triton kernel ======
        triton_batched_dense_vec_jagged_2d_matmul[self._grid](
            jagged_tensor_ptr=self._values,
            dense_ptr=self._dense,
            jagged_offset=self._offset,
            thread_block_col_size=self._thread_block_col_size,
            dense_row_stride=self._dense.stride(0),
            jagged_value_row_stride=self._values.stride(0),
            D=self._D,
            H=self._H,
            output_ptr=self._output,
        )

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        return self._output

    def _build_valid_offset(self, B, total_L, device):
        # 保证 total_L >= B（每段至少1行）
        total_L = max(total_L, B)

        offsets = [0]
        avg = total_L // B  # 每段平均长度（>=1，因total_L>=B）

        for i in range(1, B):
            seg = max(1, avg + (i % 3 - 1) * (avg // 4))
            nxt = offsets[-1] + seg
            nxt = min(nxt, total_L - (B - i))
            nxt = max(nxt, offsets[-1] + 1)
            offsets.append(nxt)

        offsets.append(total_L)
        return torch.tensor(offsets, dtype=torch.int64, device=device)
