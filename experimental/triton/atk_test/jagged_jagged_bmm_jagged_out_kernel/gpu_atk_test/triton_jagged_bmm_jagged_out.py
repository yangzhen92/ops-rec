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

import torch
import triton
import triton.language as tl


@triton.jit
def jagged_jagged_bmm_jagged_out_kernel(
    a_ptr,
    a_offset_ptr,
    b_ptr,
    b_offset_ptr,
    c_ptr,
    offsets_mn_ptr,
    max_seq_len,
    num_blocks_n,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing C = A x B.
    A has shape (sum_B(Mi), K), B has shape (K, sum_B(Ni))
    and C has shape (sum_B(Mi * Ni))
    """

    pid = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_a = tl.load(a_offset_ptr + pid_batch)
    end_a = tl.load(a_offset_ptr + pid_batch + 1)

    begin_b = tl.load(b_offset_ptr + pid_batch)
    end_b = tl.load(b_offset_ptr + pid_batch + 1)

    offset_mn = tl.load(offsets_mn_ptr + pid_batch)

    M = end_a - begin_a
    M = tl.minimum(M, max_seq_len)

    N = end_b - begin_b
    N = tl.minimum(N, max_seq_len)

    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + begin_a * stride_am
    )

    b_ptrs = (
        b_ptr
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        + begin_b * stride_bn
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        updated_offset = k + offs_k
        a = tl.load(
            a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=((updated_offset[None, :] < K) & (offs_am[:, None] < M)),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=((updated_offset[:, None] < K) & (offs_bn[None, :] < N)),
            other=0.0,
        )
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offset_mn + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_jagged_jagged_bmm_jagged_out(
    jagged_A,
    jagged_B,
    max_seq_len,
    lengths_m,
    lengths_n,
    lengths_mn,
    offsets_m,
    offsets_n,
    offsets_mn,
    allow_tf32=False,
):
    assert jagged_A.size(1) == jagged_B.size(0), "incompatible dimensions"
    assert offsets_mn.is_contiguous(), "mn offsets mush be contiguous"
    assert offsets_m.is_contiguous(), "m offsets mush be contiguous"
    assert offsets_n.is_contiguous(), "n offsets mush be contiguous"

    B = lengths_m.size(0)
    jagged_C = torch.zeros(
        (lengths_mn.sum()), device=jagged_A.device, dtype=jagged_A.dtype
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    num_blocks_m = triton.cdiv(max_seq_len, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(max_seq_len, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n, B)

    jagged_jagged_bmm_jagged_out_kernel[grid](
        jagged_A,
        offsets_m,
        jagged_B,
        offsets_n,
        jagged_C,
        offsets_mn,
        max_seq_len,
        num_blocks_n,
        jagged_A.size(1),
        jagged_A.stride(0),
        jagged_A.stride(1),
        jagged_B.stride(0),
        jagged_B.stride(1),
        allow_tf32,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    return jagged_C