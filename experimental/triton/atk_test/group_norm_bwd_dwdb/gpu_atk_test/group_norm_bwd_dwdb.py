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
import triton
import triton.language as tl


@triton.jit
def _group_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    BLOCK_N: tl.constexpr,
):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw.to(FINAL_DW.dtype.element_ty))
    tl.store(FINAL_DB + col, sum_db.to(FINAL_DB.dtype.element_ty))
