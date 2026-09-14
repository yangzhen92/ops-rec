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

# triton_jagged_reduce_sum_api.py
"""jagged_reduce_sum 的 ATK API 封装。"""

from __future__ import annotations

import torch
import triton

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from triton_jagged_reduce_sum_kernel import jagged_reduce_sum

# pylint: disable=attribute-defined-outside-init

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MAX_TOTAL_ELEMENTS = 1 << 24


def _get_device() -> str:
    """ATK 在对应设备节点执行时优先选择 NPU，否则选择 CUDA。"""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("jagged_reduce_sum requires an available NPU or CUDA device")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _build_offsets(batch_size: int, total_length: int, d: int, device: str) -> torch.Tensor:
    """按 shape 确定性生成合法 offsets，确保 GPU/NPU 使用完全相同的分段规则。"""
    if batch_size < 1:
        raise ValueError(f"B must be >= 1, got {batch_size}")
    if total_length < 0:
        raise ValueError(f"L must be >= 0, got {total_length}")

    # 大多数用例均匀划分。部分 shape 固定保留一个空段，验证 seq_offsets 相等时
    # kernel 的 fp32 零初始化和空循环分支；规则只依赖 shape，不含随机数。
    force_empty_segment = batch_size > 1 and (total_length + d) % 4 == 0
    active_segments = batch_size - 1 if force_empty_segment else batch_size
    base_length, remainder = divmod(total_length, active_segments)
    empty_segment = (total_length + d) % batch_size if force_empty_segment else -1

    offsets = [0]
    active_index = 0
    for batch_index in range(batch_size):
        if batch_index == empty_segment:
            segment_length = 0
        else:
            segment_length = base_length + int(active_index < remainder)
            active_index += 1
        offsets.append(offsets[-1] + segment_length)

    # 这些断言直接对应 kernel 的两次无 mask offset load 与 Jagged 指针偏移约束。
    # 注意：不要在 NPU tensor 上用 >= / torch.all 做校验，部分环境没有 GreaterEqual 二进制。
    assert len(offsets) == batch_size + 1
    assert offsets[0] == 0
    assert offsets[-1] == total_length
    assert all(right >= left for left, right in zip(offsets, offsets[1:]))
    return torch.tensor(offsets, device=device, dtype=torch.int64)


@register("jagged_reduce_sum")
@register("triton_jagged_reduce_sum")
class TritonJaggedReduceSumApi(BaseApi):
    """只在 init 中准备数据和 launch 元数据，__call__ 保持为纯 kernel 调用。"""

    def init_by_input_data(self, input_data: InputDataset):
        jagged_input = input_data.kwargs.get("Jagged")
        offsets_input = input_data.kwargs.get("seq_offsets")
        if not isinstance(jagged_input, torch.Tensor):
            raise ValueError("Jagged must be a tensor")
        if not isinstance(offsets_input, torch.Tensor):
            raise ValueError("seq_offsets must be a tensor")
        if jagged_input.ndim != 2:
            raise ValueError(f"Jagged must have shape [L, D], got {tuple(jagged_input.shape)}")
        if offsets_input.ndim != 1 or offsets_input.numel() < 2:
            raise ValueError("seq_offsets must have shape [B + 1] with B >= 1")
        if offsets_input.dtype != torch.int64:
            raise ValueError("seq_offsets must be int64; Generator/YAML should enforce this")
        if jagged_input.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported Jagged dtype: {jagged_input.dtype}")

        total_length, d = jagged_input.shape
        batch_size = offsets_input.numel() - 1
        if d < 1:
            raise ValueError("D must be >= 1; zero-width output cannot launch this kernel")
        if (total_length + batch_size) * d > _MAX_TOTAL_ELEMENTS:
            raise ValueError("Generator must enforce (L + B) * D <= 2^24")

        device = _get_device()
        # 原始 offsets 的随机值不参与执行，只使用其长度 B+1。由 shape 重建能保证两端一致。
        self._jagged = jagged_input.to(device).contiguous()
        self._seq_offsets = _build_offsets(batch_size, total_length, d, device)
        self._out = torch.empty((batch_size, d), device=device, dtype=self._jagged.dtype)

        # 必须与原始 backward 包装函数保持一致，否则 D=63/64/65 的编译配置和 grid 会偏离。
        self._block_d = triton.next_power_of_2(d) if d < 64 else 64
        self._grid = (batch_size, triton.cdiv(d, self._block_d))
        self._d = d
        self._device = device

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 性能计时路径中不再构造数据或计算元数据，只有同步和 kernel launch。
        _synchronize(self._device)

        jagged_reduce_sum[self._grid](
            seq_offsets=self._seq_offsets,
            Jagged=self._jagged,
            Out=self._out,
            D=self._d,
            stride_jn=self._jagged.stride(0),
            stride_ob=self._out.stride(0),
            BLOCK_D=self._block_d,
        )
        _synchronize(self._device)
        return self._out
