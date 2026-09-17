#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
# type: ignore


import torch

# torch 26.04 circular-import workaround for wrap_triton — do not remove
import torch._inductor.dependencies  # noqa: F401

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


@triton.jit
def jagged_reduce_sum(
    seq_offsets,
    Jagged,
    Out,
    D,
    stride_jn,
    stride_ob,
    BLOCK_D: tl.constexpr,
):
    """Reduce each jagged segment along its sequence dimension."""
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    Jagged += seq_start * stride_jn
    Out += off_b * stride_ob
    offs_d = off_d + tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_d
    out_ptrs = Out + offs_d
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, seq_len):
        jg = tl.load(
            jagged_ptrs,
            mask=offs_d < D,
        )
        accumulator += jg
        jagged_ptrs += stride_jn
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=offs_d < D,
    )
