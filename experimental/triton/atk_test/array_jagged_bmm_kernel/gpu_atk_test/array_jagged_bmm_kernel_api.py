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
array_jagged_bmm_kernel 算子的ATK自定义API执行文件
"""

import torch
import triton

from array_jagged_bmm_kernel import array_jagged_bmm_kernel
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def get_device():
    """获取可用的设备，优先 NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def _split_jagged(total: int, parts: int) -> list:
    """把 total 确定性变长切成 parts 份（与 generate 中实现逐字一致）。"""
    if parts <= 0:
        return []
    if total <= parts:
        return [1] * parts
    seed = (total * 1000003 + parts * 9176 + 12345) & 0x7FFFFFFF
    weights = []
    for _ in range(parts):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        weights.append((seed % 997) + 1)
    wsum = sum(weights)
    lengths = [1] * parts
    remaining = total - parts
    acc = 0
    for i in range(parts):
        add = remaining * weights[i] // wsum
        lengths[i] += add
        acc += add
    leftover = remaining - acc
    i = 0
    while leftover > 0:
        lengths[i % parts] += 1
        leftover -= 1
        i += 1
    return lengths


@register("array_jagged_bmm_kernel")
class ArrayJaggedBmmKernelApi(BaseApi):
    """array_jagged_bmm_kernel 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）：补 jagged 切分元信息，tensor 用 ATK 真实数据"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        array_A_input = input_data.kwargs.get("array_A", None)
        jagged_B_input = input_data.kwargs.get("jagged_B", None)
        offsets_am_input = input_data.kwargs.get("offsets_am", None)
        if array_A_input is None or jagged_B_input is None:
            raise ValueError("array_A or jagged_B not found")
        if offsets_am_input is None:
            raise ValueError("offsets_am not found")
        array_A = self._to_tensor(array_A_input, device)
        jagged_B = self._to_tensor(jagged_B_input, device)
        offsets_am_raw = self._to_tensor(offsets_am_input, device, torch.int32)
        B = offsets_am_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")
        total_seq = jagged_B.shape[0]
        D = jagged_B.shape[1] if jagged_B.ndim == 2 else 1
        dtype = array_A.dtype
        if total_seq < B:
            raise ValueError(f"total_seq({total_seq}) < B({B}), generator constraint violated")
        lengths_seq = _split_jagged(total_seq, B)  # = lengths_bk = lengths_cm
        lengths_am = [s * s for s in lengths_seq]
        total_am = sum(lengths_am)
        offsets_am = self._cumsum_tensor(lengths_am, device)
        offsets_bk = self._cumsum_tensor(lengths_seq, device)
        offsets_cm = offsets_bk.clone()  # cm = bk
        lengths_am_t = torch.tensor(lengths_am, dtype=torch.int32, device=device)
        lengths_bk_t = torch.tensor(lengths_seq, dtype=torch.int32, device=device)
        lengths_cm_t = lengths_bk_t.clone()
        if array_A.ndim != 1 or array_A.shape[0] != total_am:
            array_A = self._fit_1d(array_A, total_am, dtype, device)
        if jagged_B.ndim != 2 or jagged_B.shape[0] != total_seq or jagged_B.shape[1] != D:
            jagged_B = self._fit_2d(jagged_B, total_seq, D, dtype, device)
        max_seq_len = max(lengths_seq)
        self._device = device
        self._array_A = array_A.contiguous()
        self._jagged_B = jagged_B.contiguous()
        self._lengths_am = lengths_am_t
        self._lengths_bk = lengths_bk_t
        self._lengths_cm = lengths_cm_t
        self._offsets_am = offsets_am
        self._offsets_bk = offsets_bk
        self._offsets_cm = offsets_cm
        self._max_seq_len = max_seq_len

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：仅同步 + 调用 wrapper"""
        device = self._device
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        B = self._lengths_am.size(0)
        D = self._jagged_B.size(1)
        L = self._jagged_B.size(0)
        jagged_C = torch.zeros((L, D), device=self._jagged_B.device, dtype=self._jagged_B.dtype)
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
        grid = (
            triton.cdiv(D, BLOCK_SIZE_N),
            triton.cdiv(self._max_seq_len, BLOCK_SIZE_M),
            B,
        )
        array_jagged_bmm_kernel[grid](
            self._array_A,
            self._jagged_B,
            jagged_C,
            self._offsets_am,
            self._offsets_bk,
            self._offsets_cm,
            D,
            self._jagged_B.stride(0),
            self._jagged_B.stride(1),
            jagged_C.stride(0),
            jagged_C.stride(1),
            0,  # transpose
            self._max_seq_len,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            False,  # allow_tf32
        )
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        return jagged_C

    def _cumsum_tensor(self, lengths, device):
        """由 lengths 生成 offset 张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _fit_1d(self, raw, target_len, dtype, device):
        """把 raw 拟合到 [target_len]：够长截断，不够循环平铺（仅兜底极端不一致）"""
        flat = raw.reshape(-1).to(device=device, dtype=dtype)
        if flat.shape[0] < target_len:
            reps = (target_len + flat.shape[0] - 1) // flat.shape[0]
            flat = flat.repeat(reps)
        return flat[:target_len].contiguous()

    def _fit_2d(self, raw, rows, cols, dtype, device):
        """把 raw 拟合到 [rows, cols]"""
        flat = self._fit_1d(raw, rows * cols, dtype, device)
        return flat.reshape(rows, cols).contiguous()

    def _to_tensor(self, data, device, dtype=None):
        """转换数据到 tensor"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(data, dtype=dtype, device=device)
