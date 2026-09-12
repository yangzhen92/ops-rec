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
def _bwd_preprocess_do_o_dot(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    stride_ob,
    stride_ot,
    stride_od,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_t = tl.program_id(0)
    offs_t = start_t * BLOCK_T + tl.arange(0, BLOCK_T)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    do_ptrs = (
        do_ptr
        + pid_b * stride_do_b
        + offs_t[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )
    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))


@triton.jit
def _jagged_dense_flash_attention_bwd_dv_db_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ab_ptr,  # attn bias
    jagged_offsets_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbias_ptr,
    max_seq_len,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kd,
    stride_kt,
    stride_vl,
    stride_vd,
    stride_ab_b,  # attn bias stride batch
    stride_ab_l,
    stride_ab_t,
    stride_ob,
    stride_ot,
    stride_od,
    stride_dq_l,
    stride_dq_d,
    stride_dv_l,
    stride_dv_d,
    stride_db_b,
    stride_db_l,
    stride_db_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    # begin offset of the current sample
    begin = tl.load(jagged_offsets_ptr + pid_b)
    # end offset of the current sample
    end = tl.load(jagged_offsets_ptr + pid_b + 1)

    # The seq length of the current sample
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)

    if seqlen == 0:
        return

    q_start_ptr = q_ptr + begin * stride_ql
    k_start_ptr = k_ptr + pid_b * stride_kb
    ab_start_ptr = ab_ptr + pid_b * stride_ab_b
    v_start_ptr = v_ptr + begin * stride_vl
    do_start_ptr = do_ptr + pid_b * stride_do_b
    dq_start_ptr = dq_ptr + begin * stride_dq_l
    dv_start_ptr = dv_ptr + begin * stride_dv_l
    dbias_start_ptr = dbias_ptr + pid_b * stride_db_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    start_l = pid_l * BLOCK_L
    offs_l_curr = start_l + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)

    q_ptrs = (
        q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd
    )
    k_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t[None, :] * stride_kt
    v_ptrs = (
        v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd
    )

    do_ptrs = (
        do_start_ptr + offs_t[:, None] * stride_do_t + offs_d[None, :] * stride_do_d
    )

    dq = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)

    # Load a block of q into [BLOCK_L, BLOCK_D]
    q = tl.load(
        q_ptrs,
        mask=((offs_l_curr[:, None] < seqlen) & (offs_d[None, :] < BLOCK_D)),
        other=0.0,
    )
    v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

    # for start_t in range(0, T, BLOCK_T):
    start_t = 0
    while start_t < T:
        offs_t_curr = start_t + tl.arange(0, BLOCK_T)

        # Load a block of k into [BLOCK_D, BLOCK_T]
        k = tl.load(
            k_ptrs,
            mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
            other=0.0,
        )
        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]
        qk += tl.dot(q, k, allow_tf32=allow_tf32)

        ab_ptrs = (
            ab_start_ptr
            + offs_l_curr[:, None] * stride_ab_l
            + offs_t_curr[None, :] * stride_ab_t
        )

        ab = tl.load(
            ab_ptrs,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
            other=0.0,
        )

        # q*k output + attn bias
        qk = qk + ab

        # Mask out invalid positions for softmax
        qk_mask = (offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)
        qk = tl.where(qk_mask, qk, float("-inf"))

        lse_t = tl.load(
            lse_ptrs + offs_t_curr, mask=(offs_t_curr < T), other=float("inf")
        )
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dv += tl.dot(p, do, allow_tf32=allow_tf32)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        dp = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp += tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Save dbias = ds
        dbias_ptrs = (
            dbias_start_ptr
            + offs_l_curr[:, None] * stride_db_l
            + offs_t_curr[None, :] * stride_db_t
        )
        tl.store(
            dbias_ptrs,
            ds,
            mask=((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T)),
        )

        # Compute dq
        # gemm [BLOCK_L, BLOCK_T] x [BLOCK_T, BLOCK_D] -> [BLOCK_L, BLOCK_D]
        dq += tl.dot(ds, tl.trans(k), allow_tf32=allow_tf32)

        k_ptrs += BLOCK_T * stride_kt
        do_ptrs += BLOCK_T * stride_do_t
        start_t += BLOCK_T

    dq_ptrs = (
        dq_start_ptr
        + offs_l_curr[:, None] * stride_dq_l
        + offs_d[None, :] * stride_dq_d
    )
    dv_ptrs = (
        dv_start_ptr
        + offs_l_curr[:, None] * stride_dv_l
        + offs_d[None, :] * stride_dv_d
    )
    tl.store(dq_ptrs, dq, mask=(offs_l_curr[:, None] < seqlen))
    tl.store(dv_ptrs, dv, mask=(offs_l_curr[:, None] < seqlen))
