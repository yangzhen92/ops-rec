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
_add_timestamp_position_embeddings_kernel kernel 的 ATK API 封装（双标杆直测版）。
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
_NUM_TIME_BUCKETS = 2048
_TIME_BUCKET_INCREMENTS = 1.0
_TIME_BUCKET_SCALE = 1.0
_TIME_DELTA = 0.0
_BLOCK_N = 64


def _import_local_kernel():
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from _add_timestamp_position_embeddings_kernel import _add_timestamp_position_embeddings_kernel

    return _add_timestamp_position_embeddings_kernel


_add_timestamp_position_embeddings_kernel = _import_local_kernel()


def _resolve_device(tensor: torch.Tensor) -> str:
    """按输入 tensor 所在设备决定目标设备。"""
    device = tensor.device.type
    if device in ("cpu", "npu", "cuda"):
        return device
    raise RuntimeError(f"_add_timestamp_position_embeddings_kernel got unsupported input device: {device}")


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


def _compute_ts_inds(ts: torch.Tensor, offsets, time_bucket_fn: str):
    """按 kernel 相同公式计算时间桶索引（CPU 上，int64 返回）。"""
    offsets = offsets.long()
    idx = torch.arange(ts.numel(), dtype=torch.int64)
    batch = torch.searchsorted(offsets, idx, right=True) - 1
    query_time = ts[offsets[batch + 1] - 1]
    t = query_time - ts
    t = torch.where(t > 1e-6, t, torch.full_like(t, 1e-6))
    t = t / _TIME_BUCKET_INCREMENTS
    if time_bucket_fn == "log":
        t = torch.log(t)
    else:
        t = torch.sqrt(t)
    t = t * _TIME_BUCKET_SCALE
    ti = t.to(torch.int32).to(torch.int64)
    ti = torch.where(ti > 0, ti, torch.zeros_like(ti))
    ti = torch.where(ti < _NUM_TIME_BUCKETS, ti, torch.full_like(ti, _NUM_TIME_BUCKETS))
    return ti


def _compute_pos_inds(lengths, offsets, max_contextual_seq_len: int, max_pos_ind: int):
    """按 kernel 相同公式计算位置桶索引（CPU 上，int64 返回）。"""
    lengths = lengths.long()
    offsets = offsets.long()
    idx = torch.arange(lengths.sum().item(), dtype=torch.int64)
    batch = torch.searchsorted(offsets, idx, right=True) - 1
    offs_n = idx - offsets[batch]
    high_ind = torch.minimum(offs_n, lengths[batch])
    p = lengths[batch] - high_ind + max_contextual_seq_len
    p = torch.minimum(p, torch.full_like(p, max_pos_ind - 1))
    p = torch.where(offs_n < max_contextual_seq_len, offs_n, p)
    return p


@register("_add_timestamp_position_embeddings_kernel")
@register("triton_position._add_timestamp_position_embeddings_kernel")
class AddTimestampPositionEmbeddingsKernelApi(BaseApi):
    """_add_timestamp_position_embeddings_kernel kernel 的 API 封装。

    固定 BLOCK_N=64（候选最大值）；Offsets/Lengths/TS/PosEmb/TsEmb 由
    SeqEmb 与 attrs 确定性推导，保证 GPU/NPU 两端输入一致。
    """

    def init_by_input_data(self, input_data: InputDataset):
        # ── 1. 读取输入 ──
        seq_input = input_data.kwargs.get("SeqEmb")
        num_batches = int(input_data.kwargs.get("num_batches", 2))
        max_contextual_seq_len = int(input_data.kwargs.get("max_contextual_seq_len", 64))
        time_bucket_fn = str(input_data.kwargs.get("time_bucket_fn", "sqrt"))

        if seq_input is None or not isinstance(seq_input, torch.Tensor):
            raise ValueError("SeqEmb must be a tensor")
        if seq_input.ndim != 2:
            raise ValueError(f"SeqEmb must be 2D, got {seq_input.ndim}D")

        jagged_size, d = seq_input.shape
        if jagged_size == 0 or d == 0:
            raise ValueError(f"SeqEmb cannot have zero dim, got shape {seq_input.shape}")
        num_batches = max(1, num_batches)
        if num_batches > jagged_size:
            raise ValueError(
                f"num_batches={num_batches} > jagged_size={jagged_size}. "
                f"Generator should ensure jagged_size >= num_batches"
            )

        # ── 2. 统一 dtype（以 SeqEmb 为准）──
        x = seq_input
        if x.dtype == torch.float64:
            x = x.to(torch.float32).contiguous()
        if x.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported SeqEmb dtype: {x.dtype}")
        dtype = x.dtype

        # CPU 上对齐到目标 dtype，保证两端输入一致。
        in_cpu = x.cpu().to(dtype).contiguous()

        # ── 3. 确定性推导 Offsets / Lengths（均匀分段，余数放前）──
        base, rem = divmod(jagged_size, num_batches)
        lengths_cpu = torch.full((num_batches,), base, dtype=torch.int32)
        if rem:
            lengths_cpu[:rem] += 1
        offsets_cpu = torch.zeros((num_batches + 1,), dtype=torch.int32)
        torch.cumsum(lengths_cpu, dim=0, out=offsets_cpu[1:])
        max_seq_len = int(lengths_cpu.max().item())
        if max_seq_len > 4096:
            raise ValueError(
                f"max_seq_len={max_seq_len} > 4096 breaks exact-square guarantee. "
                f"Generator should ensure jagged_size <= num_batches * 4096"
            )

        # ── 4. 确定性推导 TS（完美平方间隔），保证 sqrt 桶索引精确 ──
        ts_cpu = torch.empty((jagged_size,), dtype=torch.float32)
        offs = offsets_cpu.tolist()
        lens = lengths_cpu.tolist()
        for b in range(num_batches):
            s0, s1 = offs[b], offs[b + 1]
            L = lens[b]
            C = (L - 1) ** 2
            n = torch.arange(L, dtype=torch.int64)
            diff = (L - 1 - n) ** 2
            ts_cpu[s0:s1] = (C - diff).float()

        # ── 5. 确定性构造 PosEmb / TsEmb（fp32，值域 [-1,1]）──
        max_pos_ind = max_contextual_seq_len + max_seq_len
        n_p = max_pos_ind
        n_t = _NUM_TIME_BUCKETS + 1
        row = torch.arange(n_p, dtype=torch.float32)
        col = torch.arange(d, dtype=torch.float32)
        pos_emb_cpu = ((row % 8).view(-1, 1) / 8.0 - 0.5) + ((col % 4).view(1, -1) / 8.0 - 0.25)
        pos_emb_cpu = pos_emb_cpu.contiguous()
        row_t = torch.arange(n_t, dtype=torch.float32)
        ts_emb_cpu = ((row_t % 7).view(-1, 1) / 7.0 - 0.5) + ((col % 3).view(1, -1) / 8.0 - 0.2)
        ts_emb_cpu = ts_emb_cpu.contiguous()

        # ── 6. 编译参数（固定 BLOCK_N / BLOCK_D）──
        block_d = _next_pow2(d) if d < 64 else 64
        grid = (num_batches, triton.cdiv(max_seq_len, _BLOCK_N))
        # 调优 key 参数：kernel 内部未使用，固定值即可
        autotune_max_seq_len = 1 << (max_seq_len - 1).bit_length()
        if autotune_max_seq_len > max_seq_len:
            autotune_max_seq_len //= 2

        # ── 7. 移到目标设备，分配 Out / PosInds / TsInds / NumTargets ──
        device = _resolve_device(seq_input)
        in_d = in_cpu.to(device)
        offsets_d = offsets_cpu.to(device)
        lengths_d = lengths_cpu.to(device)
        ts_d = ts_cpu.to(device)
        pos_emb_d = pos_emb_cpu.to(device)
        ts_emb_d = ts_emb_cpu.to(device)
        out = torch.empty((jagged_size, d), dtype=dtype, device=device)
        pos_inds = torch.empty((jagged_size,), dtype=torch.int32, device=device)
        ts_inds = torch.empty((jagged_size,), dtype=torch.int32, device=device)
        num_targets = torch.zeros((num_batches,), dtype=torch.int32, device=device)

        # 保存实例变量
        self._device = device
        self._jagged_size = jagged_size
        self._d = d
        self._num_batches = num_batches
        self._max_seq_len = max_seq_len
        self._max_contextual_seq_len = max_contextual_seq_len
        self._time_bucket_fn = time_bucket_fn
        self._max_pos_ind = max_pos_ind
        self._block_d = block_d
        self._grid = grid
        self._autotune_max_seq_len = autotune_max_seq_len
        self._in = in_d
        self._offsets = offsets_d
        self._lengths = lengths_d
        self._ts = ts_d
        self._pos_emb = pos_emb_d
        self._ts_emb = ts_emb_d
        self._out = out
        self._pos_inds = pos_inds
        self._ts_inds = ts_inds
        self._num_targets = num_targets
        self._stride_sn = in_d.stride(0)
        self._stride_pn = pos_emb_d.stride(0)
        self._stride_tn = ts_emb_d.stride(0)
        self._stride_on = out.stride(0)
        # CPU golden 复用（CPU tensor 副本）
        self._in_cpu = in_cpu
        self._offsets_cpu = offsets_cpu.long()
        self._lengths_cpu = lengths_cpu
        self._ts_cpu = ts_cpu
        self._pos_emb_cpu = pos_emb_cpu
        self._ts_emb_cpu = ts_emb_cpu

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._device == "cpu":
            pos_inds = _compute_pos_inds(
                self._lengths_cpu,
                self._offsets_cpu,
                self._max_contextual_seq_len,
                self._max_pos_ind,
            )
            ts_inds = _compute_ts_inds(self._ts_cpu, self._offsets_cpu, self._time_bucket_fn)
            pos_emb = self._pos_emb_cpu.index_select(0, pos_inds.to(torch.int64))
            ts_emb = self._ts_emb_cpu.index_select(0, ts_inds.to(torch.int64))
            emb_sum = (pos_emb + ts_emb).to(self._in_cpu.dtype)
            ref = self._in_cpu + emb_sum
            return ref

        _synchronize(self._device)

        # kernel 逐 token 全覆盖写 Out，无需归零
        _add_timestamp_position_embeddings_kernel[self._grid](
            SeqEmb=self._in,
            Offsets=self._offsets,
            Lengths=self._lengths,
            PosEmb=self._pos_emb,
            TsEmb=self._ts_emb,
            Out=self._out,
            TS=self._ts,
            PosInds=self._pos_inds,
            TsInds=self._ts_inds,
            NumTargets=self._num_targets,
            AUTOTUNE_MAX_SEQ_LEN=self._autotune_max_seq_len,
            D=self._d,
            num_time_buckets=_NUM_TIME_BUCKETS,
            time_bucket_increments=_TIME_BUCKET_INCREMENTS,
            time_bucket_scale=_TIME_BUCKET_SCALE,
            time_delta=_TIME_DELTA,
            max_contextual_seq_len=self._max_contextual_seq_len,
            max_pos_ind=self._max_pos_ind,
            stride_sn=self._stride_sn,
            stride_pn=self._stride_pn,
            stride_tn=self._stride_tn,
            stride_on=self._stride_on,
            TRAINING=True,
            HAS_MULTIPLE_TARGETS=False,
            INTERLEAVE_TARGETS=False,
            TIME_BUCKET_FN=self._time_bucket_fn,
            BLOCK_D=self._block_d,
            BLOCK_N=_BLOCK_N,
        )

        _synchronize(self._device)

        return self._out
