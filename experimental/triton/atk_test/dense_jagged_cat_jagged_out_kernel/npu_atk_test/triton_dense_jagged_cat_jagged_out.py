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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pyre-unsafe

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small block sizes for small sequences
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        # Medium block sizes (default baseline)
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        # Large block sizes for long sequences
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
    ],
    key=["max_seq_len"],
)
@triton.jit
def dense_jagged_cat_jagged_out_kernel(
    a_ptr,  # dense
    b_ptr,  # jagged
    c_ptr,  # jagged
    b_offsets_ptr,
    c_offsets_ptr,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    b_start = tl.load(b_offsets_ptr + pid_batch)
    b_end = tl.load(b_offsets_ptr + pid_batch + 1)
    c_start = b_start + pid_batch
    N = b_end - b_start
    N = tl.minimum(N, max_seq_len)

    a = tl.load(a_ptr + pid_batch)
    tl.store(c_ptr + c_start, a)

    offs_k = tl.arange(0, BLOCK_SIZE)
    for k in range(0, N, BLOCK_SIZE):
        b_offset = k + offs_k
        b_ptrs = b_ptr + b_start + b_offset
        b = tl.load(b_ptrs, mask=b_offset < N, other=0.0)
        tl.store(c_ptr + c_start + 1 + b_offset, b, mask=b_offset < N)
    tl.store(c_offsets_ptr + pid_batch, b_start + pid_batch)


def dense_jagged_cat_jagged_out(
    a: torch.Tensor,
    b: torch.Tensor,
    b_offsets: torch.Tensor,
    max_seq_len: int,
):
    assert a.is_contiguous()
    assert b.is_contiguous()
    assert b_offsets.is_contiguous()
    B = a.size(0)
    c = torch.zeros(b.size(0) + a.size(0), dtype=a.dtype, device=a.device)
    c_offsets = torch.empty(b_offsets.size(0), dtype=b_offsets.dtype, device=b_offsets.device)  # B + 1

    # Note: BLOCK_SIZE is now an autotune parameter, not passed explicitly
    dense_jagged_cat_jagged_out_kernel[(B,)](
        a,
        b,
        c,
        b_offsets,
        c_offsets,
        max_seq_len,
    )

    c_offsets[-1] = b_offsets[-1] + B

    return c, c_offsets
