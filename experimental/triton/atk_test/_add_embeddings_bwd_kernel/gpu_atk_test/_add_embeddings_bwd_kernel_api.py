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

"""_add_embeddings_bwd_kernel kernel 的 ATK API 封装（双标杆直测版）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
"""kernel 支持的 dtype。"""


def _import_local_kernel():
    """从同目录的 _add_embeddings_bwd_kernel.py 导入目标 kernel。
    """
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from _add_embeddings_bwd_kernel import _add_embeddings_bwd_kernel
    return _add_embeddings_bwd_kernel


_add_embeddings_bwd_kernel = _import_local_kernel()


def _resolve_device(tensor: torch.Tensor) -> str:
    """按输入 tensor 所在设备决定目标设备。
    """
    device = tensor.device.type
    if device in ("cpu", "npu", "cuda"):
        return device
    raise RuntimeError(f"_add_embeddings_bwd_kernel got unsupported input device: {device}")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@register("_add_embeddings_bwd_kernel")
@register("triton_position._add_embeddings_bwd_kernel")
class AddEmbeddingsBwdKernelApi(BaseApi):
    """_add_embeddings_bwd_kernel kernel 的 API 封装。
    """

    def init_by_input_data(self, input_data: InputDataset):
        # ── 1. 读取输入 ──
        in_input = input_data.kwargs.get("In")
        num_keys = int(input_data.kwargs.get("num_keys", 16))

        if in_input is None or not isinstance(in_input, torch.Tensor):
            raise ValueError("In must be a tensor")
        if in_input.ndim != 2:
            raise ValueError(f"In must be 2D, got {in_input.ndim}D")

        jagged_size, d = in_input.shape
        if jagged_size == 0 or d == 0:
            raise ValueError(f"In cannot have zero dim, got shape {in_input.shape}")
        num_keys = max(1, num_keys)
        if num_keys > jagged_size:
            raise ValueError(
                f"num_keys={num_keys} > jagged_size={jagged_size}. "
                f"Generator should ensure jagged_size >= num_keys"
            )

        # ── 2. 统一 dtype（以 In 为准）──
        x = in_input
        if x.dtype == torch.float64:
            x = x.to(torch.float32).contiguous()
        if x.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported In dtype: {x.dtype}")
        dtype = x.dtype

        # CPU 上对齐到目标 dtype，保证两端输入一致。
        in_cpu = x.cpu().to(dtype).contiguous()

        # ── 3. 确定性推导 KeyInds / ValueInds ──
        value_inds = torch.arange(jagged_size, dtype=torch.int32)
        key_inds = (
            torch.arange(jagged_size, dtype=torch.int64) * num_keys
        ) // jagged_size
        key_inds = key_inds.to(torch.int32)

        # ── 4. 编译参数（跳过 autotune，固定 BLOCK=128）──
        block_d = _next_pow2(d)
        block = 128
        grid = (triton.cdiv(jagged_size, block),)

        # ── 5. 移到目标设备，Out 零初始化（atomic_add 累加）──
        # golden（double_benchmark）的输入在 CPU worker 里已在 CPU 上，
        # 直接按输入 tensor 所在设备执行即可，不依赖默认卡 0。
        device = _resolve_device(in_input)
        in_d = in_cpu.to(device)
        key_d = key_inds.to(device)
        value_d = value_inds.to(device)
        out = torch.zeros((num_keys, d), dtype=dtype, device=device)

        # 保存实例变量
        self._device = device
        self._jagged_size = jagged_size
        self._d = d
        self._num_keys = num_keys
        self._block_d = block_d
        self._block = block
        self._grid = grid
        self._in = in_d
        self._key_inds = key_d
        self._value_inds = value_d
        self._out = out
        self._stride_in = in_d.stride(0)
        self._stride_on = out.stride(0)
        # autotune key 参数：kernel 内部未使用，固定值即可
        self._autotune_max_seq_len = 1024
        self._b = 1
        self._autotune_b = 1

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._device == "cpu":
            # double_benchmark 的 CPU golden：triton kernel 无法在 CPU 上运行，
            # 用 torch 参考（index_add）计算真值（与原 debug 块逻辑一致）。
            key_map = (
                torch.arange(self._jagged_size, dtype=torch.int64) * self._num_keys
            ) // self._jagged_size
            ref = torch.zeros((self._num_keys, self._d), dtype=self._in.dtype)
            ref.index_add_(0, key_map, self._in)
            return ref

        _synchronize(self._device)

        # atomic_add 累加语义要求每次调用从零开始
        self._out.zero_()

        _add_embeddings_bwd_kernel[self._grid](
            In=self._in,
            KeyInds=self._key_inds,
            ValueInds=self._value_inds,
            Out=self._out,
            AUTOTUNE_MAX_SEQ_LEN=self._autotune_max_seq_len,
            B=self._b,
            AUTOTUNE_B=self._autotune_b,
            D=self._d,
            jagged_size=self._jagged_size,
            stride_in=self._stride_in,
            stride_on=self._stride_on,
            BLOCK_D=self._block_d,
            BLOCK=self._block,
        )

        _synchronize(self._device)

        return self._out

