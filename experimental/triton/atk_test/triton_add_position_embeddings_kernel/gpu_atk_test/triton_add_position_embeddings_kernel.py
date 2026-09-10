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
from triton.runtime.autotuner import Autotuner


def triton_autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    warmup: int = 25,
    rep: int = 100,
):
    def decorator(fn):
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=None,
            post_hook=None,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
        )

    return decorator


def _add_position_embeddings_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [16, 32, 64]:
        for num_stages in [1, 2]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": BLOCK_N},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_add_position_embeddings_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _add_position_embeddings_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    scale,
    ind_offsets,
    stride_jn,
    stride_dk,
    stride_on,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Jagged has shape (sum_B(N_i), D),
    Dense has shape (K, D),
    Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    max_ind = tl.load(high_inds + off_b)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    if ind_offsets is None:
        clamped_offs_n = tl.where(offs_n >= max_ind, max_ind, offs_n)
    else:
        ind_offset = tl.load(ind_offsets + off_b)
        clamped_offs_n = tl.where(offs_n + ind_offset >= max_ind, max_ind, offs_n + ind_offset)
    offs_d = tl.arange(0, BLOCK_D)
    Jagged += seq_start.to(tl.int64) * stride_jn
    jagged_ptr_offsets = offs_n[:, None] * stride_jn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_ptrs = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    dense_ptrs = Dense + clamped_offs_n[:, None] * stride_dk + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)
        jg = tl.load(Jagged + jagged_ptr_offsets, mask=mask)
        if SCALE_JAGGED:
            jg = jg * scale
        dn = tl.load(dense_ptrs, mask=mask)
        jg += dn
        tl.store(out_ptrs, jg, mask=mask)
        dense_ptrs += BLOCK_D
        out_ptrs += BLOCK_D
        offs_d += BLOCK_D
        jagged_ptr_offsets += BLOCK_D
