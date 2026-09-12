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

    o_ptrs = o_ptr + pid_b * stride_ob + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od
    do_ptrs = do_ptr + pid_b * stride_do_b + offs_t[:, None] * stride_do_t + offs_d[None, :] * stride_do_d
    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))


@triton.jit
def _jagged_dense_flash_attention_bwd_dk_kernel(
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
    stride_dk_b,
    stride_dk_d,
    stride_dk_t,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    D,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid_t = tl.program_id(0)
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
    dk_start_ptr = dk_ptr + pid_b * stride_dk_b
    delta_ptrs = delta_ptr + pid_b * T
    lse_ptrs = lse_ptr + pid_b * T

    offs_t_curr = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    k_ptrs = k_start_ptr + offs_d[:, None] * stride_kd + offs_t_curr[None, :] * stride_kt

    do_ptrs = do_start_ptr + offs_t_curr[:, None] * stride_do_t + offs_d[None, :] * stride_do_d

    dk_ptrs = dk_start_ptr + offs_d[:, None] * stride_dk_d + offs_t_curr[None, :] * stride_dk_t

    dk = tl.zeros([BLOCK_D, BLOCK_T], dtype=tl.float32)

    # Load a block of k into [BLOCK_D, BLOCK_T]
    k = tl.load(
        k_ptrs,
        mask=((offs_t_curr[None, :] < T) & (offs_d[:, None] < BLOCK_D)),
        other=0.0,
    )

    start_l = 0
    while start_l < seqlen:
        offs_l_curr = start_l + tl.arange(0, BLOCK_L)

        # Load a block of q into [BLOCK_L, BLOCK_D]
        q_ptrs = q_start_ptr + offs_l_curr[:, None] * stride_ql + offs_d[None, :] * stride_qd

        q = tl.load(
            q_ptrs,
            mask=(offs_l_curr[:, None] < seqlen),
            other=0.0,
        )

        v_ptrs = v_start_ptr + offs_l_curr[:, None] * stride_vl + offs_d[None, :] * stride_vd

        v = tl.load(v_ptrs, mask=(offs_l_curr[:, None] < seqlen), other=0.0)

        qk = tl.zeros([BLOCK_L, BLOCK_T], dtype=tl.float32)
        # gemm [BLOCK_L, BLOCK_D] x [BLOCK_D, BLOCK_T] -> [BLOCK_L, BLOCK_T]

        qk = tl.dot(q, k, allow_tf32=allow_tf32)
        qk = tl.where((offs_l_curr[:, None] < seqlen) & (offs_t_curr[None, :] < T), qk, 0.0)

        ab_ptrs = ab_start_ptr + offs_l_curr[:, None] * stride_ab_l + offs_t_curr[None, :] * stride_ab_t

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

        lse_t = tl.load(lse_ptrs + offs_t_curr, mask=(offs_t_curr < T))
        # Perform softmax
        p = tl.exp(qk - lse_t[None, :])
        p = tl.where(qk_mask, p, 0.0)

        # Compute dv
        # Load a block of do into [BLOCK_T, BLOCK_D]
        do = tl.load(do_ptrs, mask=(offs_t_curr[:, None] < T), other=0.0)

        # Compute dp
        delta = tl.load(delta_ptrs + offs_t_curr, mask=(offs_t_curr < T))

        # gemm [BLOCK_T, BLOCK_D] x [BLOCK_D, BLOCK_L] = [BLOCK_T, BLOCK_L]
        # [BLOCK_T, BLOCK_L]^T -> [BLOCK_L, BLOCK_T]
        dp = tl.trans(tl.dot(do, tl.trans(v), allow_tf32=allow_tf32))

        # Compute ds = p * (dp - delta)
        ds = p * (dp - delta[None, :])

        # Compute dk
        # gemm [BLOCK_D, BLOCK_L] x [BLOCK_L, BLOCK_T] = [BLOCK_D, BLOCK_T]
        dk += tl.dot(tl.trans(q), ds, allow_tf32=allow_tf32)

        start_l += BLOCK_L

    tl.store(dk_ptrs, dk, mask=(offs_t_curr[None, :] < T))
