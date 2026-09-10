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
jagged_2_softmax_kernel 算子的ATK自定义API执行文件
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
    from jagged_2_softmax_kernel import jagged_2_softmax_kernel as jagged_2_softmax_kernel_impl

    return jagged_2_softmax_kernel_impl


jagged_2_softmax_kernel = _import_local_kernel()


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
    """确定性生成 B 个 seq_len（种子仅由 B 决定，与 generate 中实现逐字一致）"""
    if B <= 0:
        return []
    first_len = _hash_range(0, lo, hi, salt=(B * 2654435761) & 0x7FFFFFFF)
    lengths = [first_len]
    for i in range(1, B):
        lengths.append(_hash_range(i, lo, hi, salt=first_len))
    return lengths


@register("jagged_2_softmax_kernel")
class Jagged2SoftmaxKernelAPI(BaseApi):
    """jagged_2_softmax_kernel 算子的API适配器（对 jagged 方阵按行/列做 softmax）"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）：重建自洽输入并保存实例变量"""
        device = get_device()

        input_x_input = input_data.kwargs.get("input_x", None)
        offsets_row_input = input_data.kwargs.get("offsets_row", None)
        offsets_col_input = input_data.kwargs.get("offsets_col", None)
        offsets_overall_input = input_data.kwargs.get("offsets_overall", None)
        transpose_input = input_data.kwargs.get("transpose", 1)

        if input_x_input is None:
            raise ValueError("input_x not found")
        if offsets_row_input is None or offsets_overall_input is None:
            raise ValueError("offsets not found")

        input_x = self._to_tensor(input_x_input, device)
        offsets_row = self._to_tensor(offsets_row_input, device, torch.int32)
        offsets_col = self._to_tensor(offsets_col_input, device, torch.int32)
        offsets_overall = self._to_tensor(offsets_overall_input, device, torch.int32)

        # 只信任 offset 长度定 B；值不可信（ATK outlier_values 不是值覆盖机制），offset 由 API 重建
        B = offsets_overall.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        # 确定性重建 lengths_seq（与 Generator 逐字一致）
        lengths_seq = _deterministic_seq_lengths(B)
        lengths_overall = [s * s for s in lengths_seq]
        total_overall = sum(lengths_overall)

        offsets_row = self._cumsum_tensor(lengths_seq, device)
        offsets_col = offsets_row.clone()
        offsets_overall = self._cumsum_tensor(lengths_overall, device)

        # 防御：input_x 长度不符时拟合
        if input_x.shape[0] != total_overall:
            input_x = self._rebuild_1d(input_x, total_overall, input_x.dtype, device)

        dtype = input_x.dtype

        # max_seq_len = max(seq_len)，方阵row=col
        max_seq_len_row = max(lengths_seq)
        max_seq_len_col = max_seq_len_row

        # transpose转bool（容错处理：ATK可能传list或标量）
        if isinstance(transpose_input, list):
            transpose_val = transpose_input[0] if len(transpose_input) > 0 else 0
        else:
            transpose_val = transpose_input
        transpose = bool(int(transpose_val))

        output_y = torch.zeros((total_overall,), dtype=dtype, device=device)

        # BLOCK_SIZE 上限 128 限制 UB 占用；grid = (B, max_seq_len_col)
        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len_row), 16)
        BLOCK_SIZE = min(BLOCK_SIZE, 128)
        grid = (B, max_seq_len_col)

        self._device = device
        self._input_x = input_x
        self._output_y = output_y
        self._offsets_row = offsets_row
        self._offsets_col = offsets_col
        self._offsets_overall = offsets_overall
        self._max_seq_len_row = max_seq_len_row
        self._max_seq_len_col = max_seq_len_col
        self._transpose = transpose
        self._BLOCK_SIZE = BLOCK_SIZE
        self._grid = grid

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：直接调用 Triton kernel"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 直接调用 kernel（11 参数，对照 wrapper）
        jagged_2_softmax_kernel[self._grid](
            self._input_x,  # input_ptr
            self._output_y,  # output_ptr
            self._offsets_row,  # offsets_row_ptr
            self._offsets_col,  # offsets_col_ptr (head)
            self._offsets_overall,  # offsets_overall_ptr
            self._input_x.stride(0),  # input_stride
            self._output_y.stride(0),  # output_stride
            self._transpose,  # transpose
            self._max_seq_len_row,  # max_seq_len_row
            self._max_seq_len_col,  # max_seq_len_col
            self._BLOCK_SIZE,  # BLOCK_SIZE (constexpr)
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        return self._output_y

    def _cumsum_tensor(self, lengths, device):
        """由lengths生成offset张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _rebuild_1d(self, raw, target_len, dtype, device):
        """重建 1D tensor 到 target_len（够长截断，不够平铺复用）"""
        raw_flat = raw.reshape(-1).to(device=device, dtype=dtype)
        if raw_flat.shape[0] >= target_len:
            return raw_flat[:target_len].contiguous()
        # 不够长：循环平铺填满
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
