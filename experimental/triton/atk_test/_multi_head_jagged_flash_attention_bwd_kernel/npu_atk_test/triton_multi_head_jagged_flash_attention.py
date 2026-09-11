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


import triton
import triton.language as tl


@triton.jit
def _multi_head_jagged_flash_attention_bwd_kernel(
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
    stride_delta_h,
    stride_dq_h,
    stride_dq_m,
    stride_dq_d,
    stride_dk_h,
    stride_dk_n,
    stride_dk_d,
    stride_dv_h,
    stride_dv_n,
    stride_dv_d,
    stride_do_h,
    stride_do_m,
    stride_do_d,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    seqlen = tl.minimum(end - begin, max_seq_len)

    if seqlen == 0:
        return

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_D)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    q_ptrs = (
        q_ptr + pid_head * stride_qh + begin * stride_qm + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    k_ptrs = (
        k_ptr + pid_head * stride_kh + begin * stride_kn + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    )

    v_ptrs = (
        v_ptr + pid_head * stride_vh + begin * stride_vn + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    )

    do_ptrs = (
        do_ptr
        + pid_head * stride_do_h
        + begin * stride_do_m
        + (offs_m[:, None] * stride_do_m + offs_d[None, :] * stride_do_d)
    )

    # Load a block of K into [BLOCK_N, BLOCK_D]
    k = tl.load(k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)), other=0.0)
    # Load a block of V into [BLOCK_N, BLOCK_D]
    v = tl.load(v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)), other=0.0)

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    for begin_m in range(0, seqlen, BLOCK_M):
        offs_m_curr = begin_m + offs_m

        # Load a block of Q into [BLOCK_M, BLOCK_D]
        q = tl.load(
            q_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
            other=0.0,
        )
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_curr[:, None] < seqlen) & (offs_n[None, :] < seqlen)

        # Load a block of lse_i into [BLOCK_M]
        lse_i = tl.load(
            lse_ptr + pid_head * stride_lse_h + begin + offs_m_curr,
            mask=offs_m_curr < seqlen,
            other=float("inf"),
        )

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len

        p = p.to(do_ptr.dtype.element_ty)
        do = tl.load(
            do_ptrs,
            mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
            other=0.0,
        )

        # gemm [BLOCK_N, BLOCK_M] x [BLOCK_M, BLOCK_D] -> [BLOCK_N, BLOCK_D]
        dv += tl.dot(tl.trans(p), do, allow_tf32=allow_tf32)
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(
            delta_ptr + pid_head * stride_delta_h + begin + offs_m_curr,
            mask=offs_m_curr < seqlen,
        )
        ds = p * (dp - Di[:, None] * max_seq_len)

        # compute dk = dot(ds.T, q)
        ds = ds.to(q_ptr.dtype.element_ty)
        # gemm [BLOCK_N, BLOCK_M] x [BLOCK_M, BLOCK_D] -> [BLOCK_N, BLOCK_D]
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_do_m

    # store back dk and dv
    dk_ptrs = (
        dk_ptr
        + pid_head * stride_dk_h
        + begin * stride_dk_n
        + (offs_n[:, None] * stride_dk_n + offs_d[None, :] * stride_dk_d)
    )

    dv_ptrs = (
        dv_ptr
        + pid_head * stride_dv_h
        + begin * stride_dv_n
        + (offs_n[:, None] * stride_dv_n + offs_d[None, :] * stride_dv_d)
    )

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < seqlen)))

    # Start to compute dq

    start_m = tl.program_id(axis=0) * BLOCK_M
    offs_m_curr = start_m + tl.arange(0, BLOCK_M)

    dq_ptrs_curr = (
        dq_ptr
        + pid_head * stride_dq_h
        + begin * stride_dq_m
        + (offs_m_curr[:, None] * stride_dq_m + offs_d[None, :] * stride_dq_d)
    )

    dq_curr = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_ptrs_curr = (
        q_ptr
        + pid_head * stride_qh
        + begin * stride_qm
        + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )

    q_curr = tl.load(q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)))

    # Load a block of lse_i into [BLOCK_M]
    lse_i_curr = tl.load(
        lse_ptr + pid_head * stride_lse_h + begin + offs_m_curr,
        mask=offs_m_curr < seqlen,
    )

    do_ptrs_curr = (
        do_ptr
        + pid_head * stride_do_h
        + begin * stride_do_m
        + (offs_m_curr[:, None] * stride_do_m + offs_d[None, :] * stride_do_d)
    )

    # Load do
    do_curr = tl.load(do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)))
    Di_curr = tl.load(
        delta_ptr + pid_head * stride_delta_h + begin + offs_m_curr,
        mask=offs_m_curr < seqlen,
    )

    block_start = 0
    while block_start < seqlen:
        offs_n_curr = block_start + tl.arange(0, BLOCK_N)

        k_ptrs_curr = (
            k_ptr
            + pid_head * stride_kh
            + begin * stride_kn
            + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs_curr = (
            v_ptr
            + pid_head * stride_vh
            + begin * stride_vn
            + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )

        k_curr = tl.load(k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < seqlen)))
        v_curr = tl.load(v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < seqlen)))

        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < seqlen) & (offs_n_curr[None, :] < seqlen)

        # Perform softmax
        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        # compute dp = dot(do, v.T)
        # gemm [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)

        # compute ds = p * (dp - delta[:, None])
        ds_curr = p_curr * (dp_curr - Di_curr[:, None] * max_seq_len)

        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        # compute dq = dot(ds, k)
        # gemm [BLOCK_M, BLOCK_N] x [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_N

    tl.store(
        dq_ptrs_curr,
        dq_curr,
        mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < seqlen)),
    )
