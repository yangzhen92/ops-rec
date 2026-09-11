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
def _jagged_flash_attention_bwd_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    offset_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    stride_dqm,
    stride_dqd,
    stride_dkn,
    stride_dkd,
    stride_dvn,
    stride_dvd,
    stride_dom,
    stride_dod,
    max_seq_len,
    D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    M = tl.minimum(end - begin, max_seq_len)

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    q_ptrs = (
        q_ptr
        + begin * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    k_ptrs = (
        k_ptr
        + begin * stride_kn
        + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    )

    v_ptrs = (
        v_ptr
        + begin * stride_vn
        + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    )

    do_ptrs = (
        do_ptr
        + begin * stride_dom
        + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load K and V
    k = tl.load(k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    v = tl.load(v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    for begin_m in range(0, M, BLOCK_SIZE_M):
        offs_m_temp = begin_m + offs_m

        # Load Q
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        q = tl.load(q_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_temp[:, None] < M) & (offs_n[None, :] < M)

        # Load lse_i
        lse_i = tl.load(lse_ptr + offs_m_temp + begin, mask=offs_m_temp < M)

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len
        p_masked = p

        attn_mask = None
        if use_mask:
            attn_mask = offs_m_temp[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p_masked = tl.where(attn_mask > 0, p, 0.0)

        p_masked = p_masked.to(do_ptr.dtype.element_ty)
        do = tl.load(do_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        dv += tl.dot(tl.trans(p_masked), do, allow_tf32=allow_tf32)
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(delta_ptr + offs_m_temp + begin, mask=offs_m_temp < M)
        dp_masked = dp
        if use_mask:
            dp_masked = tl.where(attn_mask > 0, dp, 0.0)

        ds = p * (dp_masked - Di[:, None] * max_seq_len)

        # compute dk = dot(ds.T, q)
        ds = ds.to(q_ptr.dtype.element_ty)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_SIZE_M * stride_qm
        do_ptrs += BLOCK_SIZE_M * stride_dom

    # store back dk and dv
    dk_ptrs = (
        dk_ptr
        + begin * stride_dkn
        + (offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    )

    dv_ptrs = (
        dv_ptr
        + begin * stride_dvn
        + (offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    )

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    start_m = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_m_curr = start_m + tl.arange(0, BLOCK_SIZE_N)

    dq_ptrs_curr = (
        dq_ptr
        + begin * stride_dqm
        + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
    )

    dq_curr = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    q_ptrs_curr = (
        q_ptr
        + begin * stride_qm
        + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    q_curr = tl.load(
        q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )

    # Load lse_i
    lse_i_curr = tl.load(lse_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    do_ptrs_curr = (
        do_ptr
        + begin * stride_dom
        + (offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dod)
    )

    # Load do
    do_curr = tl.load(
        do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )
    Di_curr = tl.load(delta_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    # When computing dV, we want to compute [BLOCK_SIZE_N] rows of dV.
    # Therefore, the loop's block size is BLOCK_SIZE_M instead of BLOCK_SIZE_N.
    block_start = 0
    while block_start < M:
        offs_n_curr = block_start + tl.arange(0, BLOCK_SIZE_M)

        k_ptrs_curr = (
            k_ptr
            + begin * stride_kn
            + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs_curr = (
            v_ptr
            + begin * stride_vn
            + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )

        k_curr = tl.load(
            k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )
        v_curr = tl.load(
            v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M))
        )

        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < M) & (offs_n_curr[None, :] < M)

        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        # compute dp = dot(v, do)
        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)
        dp_curr_masked = dp_curr

        # compute ds = p * (dp - delta[:, None])
        if use_mask:
            attn_mask = offs_m_curr[:, None] - offs_n_curr[None, :]
            attn_mask = tl.where(mn_mask_curr, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            dp_curr_masked = tl.where(attn_mask > 0, dp_curr, 0.0)

        ds_curr = p_curr * (dp_curr_masked - Di_curr[:, None] * max_seq_len)

        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_SIZE_M

    tl.store(
        dq_ptrs_curr, dq_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M))
    )