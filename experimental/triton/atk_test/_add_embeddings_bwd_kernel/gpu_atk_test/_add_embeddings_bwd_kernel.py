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

import triton
import triton.language as tl


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
