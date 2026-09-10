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
jagged_2_softmax_backward_kernel 算子的ATK自定义API执行文件
"""

import os
import sys

import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _import_local_kernel():
    """从同目录导入算子 kernel"""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from jagged_2_softmax_backward_kernel import (
        jagged_2_softmax_backward_kernel as jagged_2_softmax_backward_kernel_impl,
    )

    return jagged_2_softmax_backward_kernel_impl


jagged_2_softmax_backward_kernel = _import_local_kernel()


def get_device():
    """检测可用设备（NPU优先）"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _hash_range(i, lo, hi, salt=0):
    """确定性 LCG 哈希：把索引 i 映射到 [lo, hi]（与 generate 中实现逐字一致）"""
    x = ((i + 1) * 2654435761 + salt * 40503) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0x7FFFFFFF
    return lo + x % (hi - lo + 1)


def _deterministic_seq_lengths(B, lo=16, hi=256):
    """确定性生成 B 个 seq_len（种子仅由 B 决定，与 generate 及 forward 逐字一致）"""
    if B <= 0:
        return []
    first_len = _hash_range(0, lo, hi, salt=(B * 2654435761) & 0x7FFFFFFF)
    lengths = [first_len]
    for i in range(1, B):
        lengths.append(_hash_range(i, lo, hi, salt=first_len))
    return lengths


@register("jagged_2_softmax_backward_kernel")
class Jagged2SoftmaxBackwardKernelAPI(BaseApi):
    """jagged_2_softmax_backward_kernel 算子的API适配器（softmax backward，计算输入梯度）"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）：重建自洽输入并保存实例变量"""
        device = get_device()

        grad_output_input = input_data.kwargs.get("grad_output", None)
        softmax_output_input = input_data.kwargs.get("softmax_output", None)
        offsets_overall_input = input_data.kwargs.get("offsets_overall", None)
        transpose_input = input_data.kwargs.get("transpose", 1)

        if grad_output_input is None or softmax_output_input is None:
            raise ValueError("grad_output or softmax_output not found")
        if offsets_overall_input is None:
            raise ValueError("offsets_overall not found")

        grad_output_raw = self._to_tensor(grad_output_input, device)
        softmax_output_raw = self._to_tensor(softmax_output_input, device)
        offsets_overall_raw = self._to_tensor(offsets_overall_input, device, torch.int32)

        # 只信任 offset 长度定 B；值不可信（ATK outlier_values 不是值覆盖机制），offset 由 API 重建
        B = offsets_overall_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        dtype = grad_output_raw.dtype

        # 确定性重建 lengths_seq（与 Generator 逐字一致）
        lengths_seq = _deterministic_seq_lengths(B)
        lengths_overall = [s * s for s in lengths_seq]
        total_overall = sum(lengths_overall)

        offsets_row = self._cumsum_tensor(lengths_seq, device)
        offsets_col = offsets_row.clone()
        offsets_overall = self._cumsum_tensor(lengths_overall, device)

        # 重建 grad_output 和 softmax_output 到相同大小（关键：同shape同dtype）
        grad_output = self._rebuild_1d(grad_output_raw, total_overall, dtype, device)
        softmax_output = self._rebuild_1d(softmax_output_raw, total_overall, dtype, device)

        # max_seq_len = max(seq_len)，方阵row=col
        max_seq_len_row = max(lengths_seq)
        max_seq_len_col = max_seq_len_row

        # transpose转bool（容错处理：ATK可能传list或标量）
        if isinstance(transpose_input, list):
            transpose_val = transpose_input[0] if len(transpose_input) > 0 else 0
        else:
            transpose_val = transpose_input
        transpose = bool(int(transpose_val))

        grad_input = torch.zeros((total_overall,), dtype=dtype, device=device)

        def _grid(META):
            return (B, max_seq_len_col)

        # BLOCK_SIZE 显式传入（正确性参数，BLOCK_SIZE < N 会算错）；上限 128 限制 UB 占用
        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len_row), 16)
        BLOCK_SIZE = min(BLOCK_SIZE, 128)

        self._device = device
        self._grad_output = grad_output
        self._softmax_output = softmax_output
        self._grad_input = grad_input
        self._offsets_row = offsets_row
        self._offsets_col = offsets_col
        self._offsets_overall = offsets_overall
        self._max_seq_len_row = max_seq_len_row
        self._max_seq_len_col = max_seq_len_col
        self._transpose = transpose
        self._grid = _grid
        self._BLOCK_SIZE = BLOCK_SIZE  # 正确性参数，显式传入

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：直接调用 Triton kernel"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 关键字参数调用；BLOCK_SIZE 显式传入（正确性参数）
        jagged_2_softmax_backward_kernel[self._grid](
            grad_output_ptr=self._grad_output,
            softmax_output_ptr=self._softmax_output,
            grad_input_ptr=self._grad_input,
            offsets_row_ptr=self._offsets_row,
            offsets_col_ptr=self._offsets_col,
            offsets_overall_ptr=self._offsets_overall,
            grad_output_stride=self._grad_output.stride(0),
            softmax_output_stride=self._softmax_output.stride(0),
            grad_input_stride=self._grad_input.stride(0),
            transpose=self._transpose,
            max_seq_len_row=self._max_seq_len_row,
            max_seq_len_col=self._max_seq_len_col,
            BLOCK_SIZE=self._BLOCK_SIZE,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._grad_input

    def _cumsum_tensor(self, lengths, device):
        """由lengths生成offset张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _rebuild_1d(self, raw, target_len, dtype, device):
        """重建1D tensor到target_len长度"""
        raw_flat = raw.reshape(-1).to(device=device, dtype=dtype)
        if raw_flat.shape[0] >= target_len:
            return raw_flat[:target_len].contiguous()
        reps = (target_len + raw_flat.shape[0] - 1) // raw_flat.shape[0]
        tiled = raw_flat.repeat(reps)[:target_len]
        return tiled.contiguous()

    def _to_tensor(self, data, device, dtype=None):
        """转换数据到tensor"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return torch.tensor(data, dtype=dtype, device=device)
        return torch.tensor(data, device=device)
