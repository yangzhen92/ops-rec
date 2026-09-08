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
    # 每个 program 负责 BLOCK 个连续行,把 In 各行按 KeyInds 分组累加进 Out。
    # 行块内同一 key 的梯度先在本地累加,key 切换或行块结束时一次 atomic_add 写回。
    pid = tl.program_id(0)
    first_row = pid * BLOCK
    lane = tl.arange(0, BLOCK_D)
    col_ok = lane < D

    # 用首行初始化:grid 由 cdiv(jagged_size, BLOCK) 生成,首行必在范围内。
    src_row = tl.load(ValueInds + first_row)
    grp_key = tl.load(KeyInds + first_row)
    acc = tl.load(In + src_row.to(tl.int64) * stride_in + lane, mask=col_ok)

    for step in range(1, BLOCK):
        ridx = first_row + step
        if ridx < jagged_size:
            src_row = tl.load(ValueInds + ridx)
            x = tl.load(In + src_row.to(tl.int64) * stride_in + lane, mask=col_ok)
            k = tl.load(KeyInds + ridx)
            if k == grp_key:
                acc += x
            else:
                dst = Out + grp_key.to(tl.int64) * stride_on + lane
                tl.atomic_add(dst, acc.to(Out.dtype.element_ty), mask=col_ok, sem="relaxed")
                grp_key = k
                acc = x

    # 写回最后一个分组。
    dst = Out + grp_key.to(tl.int64) * stride_on + lane
    tl.atomic_add(dst, acc.to(Out.dtype.element_ty), mask=col_ok, sem="relaxed")
