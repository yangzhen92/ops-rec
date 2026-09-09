#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
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
from typing import List
import triton
import triton.language as tl


def get_autotune_config() -> List[triton.Config]:
    return [
        triton.Config({"BLOCK_N": 16, "multibuffer": True}),
        triton.Config({"BLOCK_N": 16, "multibuffer": True}),
        triton.Config({"BLOCK_N": 32, "multibuffer": True}),
        triton.Config({"BLOCK_N": 32, "multibuffer": True}),
        triton.Config({"BLOCK_N": 32, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "multibuffer": True}),
    ]


@triton.autotune(configs=get_autotune_config(), key=["AUTOTUNE_MAX_SEQ_LEN"])
@triton.jit
def _add_timestamp_position_embeddings_kernel(
    SeqEmb,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Out,
    TS,
    PosInds,
    TsInds,
    NumTargets,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    stride_sn,
    stride_pn,
    stride_tn,
    stride_on,
    TRAINING: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    seq_emb_offsets = offs_n[:, None] * stride_sn + offs_d[None, :]
    SeqEmb += seq_start.to(tl.int64) * stride_sn
    mask_n = offs_n < seq_len
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    if TRAINING:
        tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    if TRAINING:
        tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_offsets = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)
        seq_emb = tl.load(SeqEmb + seq_emb_offsets, mask=mask)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        tl.store(out_offsets, seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype), mask=mask)
        seq_emb_offsets += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D
