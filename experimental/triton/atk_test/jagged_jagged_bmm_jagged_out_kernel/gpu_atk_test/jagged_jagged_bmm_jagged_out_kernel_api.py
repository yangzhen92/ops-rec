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
"""jagged_jagged_bmm_jagged_out_kernel 算子的 ATK 自定义 API。"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from triton_jagged_bmm_jagged_out import triton_jagged_jagged_bmm_jagged_out


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


def _split_jagged(total: int, parts: int):
    """将 total 确定性地切分为 parts 个正整数长度。"""
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


@register("jagged_jagged_bmm_jagged_out_kernel")
class JaggedJaggedBmmJaggedOutKernelApi(BaseApi):
    """jagged_jagged_bmm_jagged_out_kernel 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）：补 jagged 切分元信息，tensor 用 ATK 真实数据"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        jagged_A_input = input_data.kwargs.get("jagged_A", None)
        jagged_B_input = input_data.kwargs.get("jagged_B", None)
        offsets_m_input = input_data.kwargs.get("offsets_m", None)

        if jagged_A_input is None or jagged_B_input is None:
            raise ValueError("jagged_A or jagged_B not found")
        if offsets_m_input is None:
            raise ValueError("offsets_m not found")

        # 统一使用 fp32 执行，避免 NPU 上 tl.dot 的输入被提升为 fp64。
        if isinstance(jagged_A_input, torch.Tensor):
            self._orig_dtype = jagged_A_input.dtype
        else:
            self._orig_dtype = torch.float32

        jagged_A = self._to_tensor(jagged_A_input, device, dtype=torch.float32)
        jagged_B = self._to_tensor(jagged_B_input, device, dtype=torch.float32)
        offsets_m_raw = self._to_tensor(offsets_m_input, device, torch.int32)

        B = offsets_m_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        total_m = jagged_A.shape[0]
        K = jagged_A.shape[1] if jagged_A.ndim == 2 else 1

        if total_m < B:
            raise ValueError(f"total_m({total_m}) < B({B}), generator constraint violated")

        lengths_m = _split_jagged(total_m, B)
        lengths_n = list(lengths_m)
        lengths_mn = [m * n for m, n in zip(lengths_m, lengths_n)]

        total_n = sum(lengths_n)

        # wrapper 要求 jagged_B 的 shape 为 [K, total_n]。
        if jagged_B.ndim != 2 or jagged_B.shape[0] != K or jagged_B.shape[1] != total_n:
            jagged_B = self._fit_2d(jagged_B, K, total_n, jagged_A.dtype, device)

        offsets_m = self._cumsum_tensor(lengths_m, device)
        offsets_n = self._cumsum_tensor(lengths_n, device)
        offsets_mn = self._cumsum_tensor(lengths_mn, device)

        lengths_m_t = torch.tensor(lengths_m, dtype=torch.int32, device=device)
        lengths_n_t = torch.tensor(lengths_n, dtype=torch.int32, device=device)
        lengths_mn_t = torch.tensor(lengths_mn, dtype=torch.int32, device=device)

        max_seq_len = max(lengths_m + lengths_n)

        self._device = device
        self._jagged_A = jagged_A.contiguous()
        self._jagged_B = jagged_B.contiguous()
        self._offsets_m = offsets_m
        self._offsets_n = offsets_n
        self._offsets_mn = offsets_mn
        self._lengths_m = lengths_m_t
        self._lengths_n = lengths_n_t
        self._lengths_mn = lengths_mn_t
        self._max_seq_len = max_seq_len

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：仅同步 + 调用 wrapper"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        jagged_C = triton_jagged_jagged_bmm_jagged_out(
            self._jagged_A,
            self._jagged_B,
            self._max_seq_len,
            self._lengths_m,
            self._lengths_n,
            self._lengths_mn,
            self._offsets_m,
            self._offsets_n,
            self._offsets_mn,
            allow_tf32=False,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and jagged_C.dtype != orig_dtype:
            jagged_C = jagged_C.to(orig_dtype)
        return jagged_C

    def _cumsum_tensor(self, lengths, device):
        """由 lengths 生成 offset 张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _fit_2d(self, raw, rows, cols, dtype, device):
        """把 raw 拟合到 [rows, cols]：够长截断，不够循环平铺（仅兜底极端不一致）"""
        flat = raw.reshape(-1).to(device=device, dtype=dtype)
        need = rows * cols
        if flat.shape[0] < need:
            reps = (need + flat.shape[0] - 1) // flat.shape[0]
            flat = flat.repeat(reps)
        return flat[:need].reshape(rows, cols).contiguous()

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
