#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# This file was copied from NVIDIA/recsys-examples
# (examples/hstu/ops/triton_ops/triton_position.py).
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


def bwd_pre_hook(nargs):
    nargs["Out"].zero_()


def get_autotune_config() -> List[triton.Config]:
    """生成 (BLOCK, num_warps) 组合的 triton.Config 列表。

    每个 config 都带 pre_hook=bwd_pre_hook：autotune 每次跑 kernel 前先把
    Out 清零，保证 atomic_add 累加从零开始（scatter-add 正确性前提）。

    不加入 multibuffer: True 的原因：本 kernel 是逐元素、数据相关分支 + atomic_add 的 scatter-add，无可流水化
    的 tile-load 循环，auto-multi-buffer 也无收益，故不加。
    """
    return [
        triton.Config({"BLOCK": 32}, num_warps=2, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2, pre_hook=bwd_pre_hook),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=2, pre_hook=bwd_pre_hook),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "AUTOTUNE_B", "D"],
)
@triton.jit
def _add_embeddings_bwd_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    B,
    AUTOTUNE_B,
    D,
    jagged_size,
    stride_in,
    stride_on,
    BLOCK_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    key_ind = -1
    key_ind = key_ind.to(KeyInds.dtype.element_ty)  # pyre-ignore[16]  # pylint: disable=no-member
    accumulator = tl.zeros((BLOCK_D,), dtype=In.dtype.element_ty)
    for off_i in range(0, BLOCK):
        off = off_block * BLOCK + off_i
        if off < jagged_size:
            value_ind = tl.load(ValueInds + off)
            in_offset = In + value_ind.to(tl.int64) * stride_in
            jagged_in = tl.load(in_offset + offs_d, mask=mask_d)
            key_ind_new = tl.load(KeyInds + off)
            if key_ind == key_ind_new:
                accumulator += jagged_in
            else:
                if key_ind >= 0:
                    out_offset = Out + key_ind.to(tl.int64) * stride_on
                    tl.atomic_add(
                        out_offset + offs_d,
                        accumulator.to(Out.dtype.element_ty),
                        mask=mask_d,
                        sem="relaxed",
                    )
                key_ind = key_ind_new
                accumulator = jagged_in
    if key_ind >= 0:
        out_offset = Out + key_ind.to(tl.int64) * stride_on
        tl.atomic_add(
            out_offset + offs_d,
            accumulator.to(Out.dtype.element_ty),
            mask=mask_d,
            sem="relaxed",
        )
