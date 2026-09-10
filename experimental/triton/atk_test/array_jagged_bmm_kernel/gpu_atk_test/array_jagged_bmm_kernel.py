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
# pyre-unsafe
# 源自 triton_jagged_bmm_jagged_out.py 的 array_jagged_bmm_kernel

import triton
import triton.language as tl


@triton.jit
def array_jagged_bmm_kernel(
    a_ptr,  # 1D array
    b_ptr,  # jagged matrix
    c_ptr,  # output, jagged matrix
    a_offsets_ptr,
    b_offsets_ptr,
    c_offsets_ptr,
    D,  # emb dimension
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    transpose,  # one if a is transpose, otherwise zero
    max_seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    batch_offset_am = tl.load(a_offsets_ptr + pid_batch)
    batch_offset_bk = tl.load(b_offsets_ptr + pid_batch)
    batch_offset_cm = tl.load(c_offsets_ptr + pid_batch)

    # calculate M, N, K
    batch_K = tl.load(b_offsets_ptr + pid_batch + 1) - batch_offset_bk  # b [batch_K, D]
    batch_M = tl.load(c_offsets_ptr + pid_batch + 1) - batch_offset_cm

    # use uncapped seq length to determine strides of a
    stride_am = batch_M * (1 - transpose) + 1 * transpose
    stride_ak = batch_M * transpose + 1 * (1 - transpose)

    # truncate seq length
    batch_K = tl.minimum(batch_K, max_seq_len)
    batch_M = tl.minimum(batch_M, max_seq_len)

    if batch_K == 0:
        return

    batch_N = D

    # c [batch_M, D] boundary check
    if pid_m * BLOCK_SIZE_M >= batch_M or pid_n * BLOCK_SIZE_N >= batch_N:
        return

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % batch_M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % batch_N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + batch_offset_am + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_offset_bk * stride_bk + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(batch_K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < batch_K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < batch_K - k * BLOCK_SIZE_K, other=0.0)
        c += tl.dot(a, b, allow_tf32=allow_tf32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * batch_offset_cm + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < batch_M) & (offs_cn[None, :] < batch_N)
    tl.store(c_ptrs, c, mask=c_mask)
