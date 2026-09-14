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
def _multi_head_jagged_flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offset_ptr,
    o_ptr,
    lse_i_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    stride_lse_h,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if pid_m * BLOCK_M >= seqlen:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    mi = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    li = tl.zeros([BLOCK_M], dtype=tl.float32)
    for j in range(0, seqlen, BLOCK_N):
        offs_n = tl.arange(0, BLOCK_N) + j
        q_ptrs = (
            q_ptr
            + pid_head * stride_qh
            + begin * stride_qm
            + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        )

        k_ptrs = (
            k_ptr
            + pid_head * stride_kh
            + begin * stride_kn
            + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        )

        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for d in range(0, D, BLOCK_D):
            curr_d = d + offs_d

            # Load a block of q into [BLOCK_M, BLOCK_D]
            q = tl.load(
                q_ptrs,
                # pyre-fixme[16]: `int` has no attribute `__getitem__`.
                mask=((curr_d[None, :] < D) & (offs_m[:, None] < seqlen)),
                other=0.0,
            )

            # Load a block of k into [BLOCK_D, BLOCK_N]
            k = tl.load(
                k_ptrs,
                mask=((curr_d[:, None] < D) & (offs_n[None, :] < seqlen)),
                other=0.0,
            )

            # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            qk += tl.dot(q, k, allow_tf32=allow_tf32)

            q_ptrs += BLOCK_D * stride_qd
            k_ptrs += BLOCK_D * stride_kd

        mi_new = tl.maximum(tl.max(qk, axis=1), mi)
        # Add the correct mask here
        mn_mask = (offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        p = tl.exp(qk - mi_new[:, None])
        p = tl.where(mn_mask, p, 0.0)

        lij_hat = tl.sum(p, axis=1)
        alpha = tl.exp(mi - mi_new)

        li = alpha * li + lij_hat
        acc = alpha[:, None] * acc

        # Load V into block [BLOCK_N, BLOCK_D]
        v_ptrs = (
            v_ptr
            + pid_head * stride_vh
            + begin * stride_vn
            + (offs_d[None, :] * stride_vd + offs_n[:, None] * stride_vn)
        )
        v = tl.load(
            v_ptrs,
            mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)),
            other=0.0,
        )

        p /= max_seq_len

        p = p.to(v_ptr.dtype.element_ty)
        # gemm [BLOCK_M, BLOCK_N] x [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc += tl.dot(p, v, allow_tf32=allow_tf32)
        mi = mi_new

    lse_i = mi + tl.math.log(li)
    lse_i_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_i_ptrs = lse_i_ptr + pid_head * stride_lse_h + begin + lse_i_offsets

    tl.store(lse_i_ptrs, lse_i, mask=lse_i_offsets < seqlen)

    acc = acc / li[:, None]

    # Store O
    o_ptrs = o_ptr + (
        pid_head * stride_oh + begin * stride_om + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc, mask=o_mask)
