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

# jagged_2_softmax_backward_kernel_api.py
"""
jagged_2_softmax_backward_kernel 算子的 ATK API 适配器

关键点：
1. API 内重建 offset（不信任 ATK 传入值）
2. 确保 grad_output 和 softmax_output 同 shape 同 dtype
3. NPU 不支持 fp64，输入统一归一到 fp32
4. BLOCK_SIZE 必须 >= N，否则 kernel 单块整行归约会静默算错
"""
import os
import sys

import torch
import triton
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# ====== 导入同目录 kernel ======
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from jagged_2_softmax_backward_kernel import jagged_2_softmax_backward_kernel


# kernel 支持的 dtype（NPU 不支持 fp64）
_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def get_device():
    """检测可用设备（NPU 优先）"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _hash_range(i, lo, hi, salt=0):
    x = ((i + 1) * 2654435761 + salt * 40503) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0x7FFFFFFF
    return lo + x % (hi - lo + 1)


def _deterministic_seq_lengths(B, lo=16, hi=256):
    if B <= 0:
        return []
    first_len = _hash_range(0, lo, hi, salt=(B * 2654435761) & 0x7FFFFFFF)
    lengths = [first_len]
    for i in range(1, B):
        lengths.append(_hash_range(i, lo, hi, salt=first_len))
    return lengths


@register("jagged_2_softmax_backward_kernel")
class Jagged2SoftmaxBackwardKernelAPI(BaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        device = get_device()

        # ====== 获取输入 ======
        grad_output_input = input_data.kwargs.get("grad_output", None)
        softmax_output_input = input_data.kwargs.get("softmax_output", None)
        offsets_overall_input = input_data.kwargs.get("offsets_overall", None)
        transpose_input = input_data.kwargs.get("transpose", 1)

        if grad_output_input is None or softmax_output_input is None:
            raise ValueError("grad_output or softmax_output not found")
        if offsets_overall_input is None:
            raise ValueError("offsets_overall not found")

        # ====== 转换到目标设备 ======
        grad_output_raw = self._to_tensor(grad_output_input, device)
        softmax_output_raw = self._to_tensor(softmax_output_input, device)
        offsets_overall_raw = self._to_tensor(offsets_overall_input, device, torch.int32)

        # ====== dtype 归一化：NPU 不支持 fp64，强制转 fp32 ======
        if grad_output_raw.dtype == torch.float64:
            grad_output_raw = grad_output_raw.to(torch.float32).contiguous()
        if softmax_output_raw.dtype == torch.float64:
            softmax_output_raw = softmax_output_raw.to(torch.float32).contiguous()

        if grad_output_raw.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported grad_output dtype: {grad_output_raw.dtype}"
            )
        if softmax_output_raw.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported softmax_output dtype: {softmax_output_raw.dtype}"
            )

        # 两端 dtype 对齐（以 grad_output 为准）
        if softmax_output_raw.dtype != grad_output_raw.dtype:
            softmax_output_raw = softmax_output_raw.to(grad_output_raw.dtype)

        dtype = grad_output_raw.dtype

        # ====== 从 offsets 提取 B ======
        # 只信任 offset 的 shape（长度）来定 B；绝不信任 offset 的 value。
        B = offsets_overall_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        # 确定性重建 lengths_seq（与 Generator 逐字一致，且与 forward 同源）
        lengths_seq = _deterministic_seq_lengths(B)

        # lengths_overall = seq_len²，total = Σseq²（有界：seq_len∈[16,256]）
        lengths_overall = [s * s for s in lengths_seq]
        total_overall = sum(lengths_overall)

        # 重建三个自洽的 offset（kernel 消费的是 offset 的 value）
        offsets_row = self._cumsum_tensor(lengths_seq, device)
        offsets_col = offsets_row.clone()
        offsets_overall = self._cumsum_tensor(lengths_overall, device)

        # 重建 grad_output 和 softmax_output 到相同大小（同 shape 同 dtype）
        grad_output = self._rebuild_1d(grad_output_raw, total_overall, dtype, device)
        softmax_output = self._rebuild_1d(
            softmax_output_raw, total_overall, dtype, device
        )

        # max_seq_len = max(seq_len)，方阵 row=col
        max_seq_len_row = max(lengths_seq)
        max_seq_len_col = max_seq_len_row

        # transpose 转 bool（容错处理：ATK 可能传 list 或标量）
        if isinstance(transpose_input, list):
            transpose_val = transpose_input[0] if len(transpose_input) > 0 else 0
        else:
            transpose_val = transpose_input
        transpose = bool(int(transpose_val))

        # ====== 创建输出 ======
        grad_input = torch.zeros((total_overall,), dtype=dtype, device=device)

        # ====== grid 与原生一致 ======
        def _grid(META):
            return (B, max_seq_len_col)

        BLOCK_SIZE = max(triton.next_power_of_2(max_seq_len_row), 16)

        # ====== 保存实例变量 ======
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
        self._BLOCK_SIZE = BLOCK_SIZE

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：直接调用 Triton kernel"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

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
        """由 lengths 生成 offset 张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for l in lengths:
            offsets.append(offsets[-1] + l)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _rebuild_1d(self, raw, target_len, dtype, device):
        """重建 1D tensor 到 target_len 长度"""
        raw_flat = raw.to(device=device, dtype=dtype).reshape(-1)
        if raw_flat.shape[0] >= target_len:
            return raw_flat[:target_len].contiguous()
        reps = (target_len + raw_flat.shape[0] - 1) // raw_flat.shape[0]
        tiled = raw_flat.repeat(reps)[:target_len]
        return tiled.contiguous()

    def _to_tensor(self, data, device, dtype=None):
        """转换数据到 tensor"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return torch.tensor(data, dtype=dtype, device=device)
        return torch.tensor(data, device=device)
