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

import triton
import triton.language as tl


@triton.jit
def jagged_flash_attention_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    offset_ptr,
    o_ptr,
    lse_i_ptr,
    stride_qm,
    stride_qd,
    stride_kd,
    stride_kn,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    NEXT_D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)
    seqlen = end - begin
    seqlen = tl.minimum(seqlen, max_seq_len)
    if pid_m * BLOCK_SIZE_M >= seqlen:
        return
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_nextd = tl.arange(0, NEXT_D)
    acc = tl.zeros([BLOCK_SIZE_M, NEXT_D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for j in range(0, seqlen, BLOCK_SIZE_N):
        offs_n = tl.arange(0, BLOCK_SIZE_N) + j
        q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd) + begin * stride_qm
        k_ptrs = k_ptr + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn) + begin * stride_kn
        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for d in range(0, D, BLOCK_SIZE_D):
            updated_offset = d + offs_d
            q = tl.load(
                q_ptrs,
                # pyre-fixme[16]: `int` has no attribute `__getitem__`.
                mask=((updated_offset[None, :] < D) & (offs_m[:, None] < seqlen)),
                other=0.0,
            )
            k = tl.load(
                k_ptrs,
                mask=((updated_offset[:, None] < D) & (offs_n[None, :] < seqlen)),
                other=0.0,
            )
            qk += tl.dot(q, k, allow_tf32=allow_tf32)
            q_ptrs += BLOCK_SIZE_D * stride_qd
            k_ptrs += BLOCK_SIZE_D * stride_kd
        m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
        mn_mask = (offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(mn_mask, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_ptrs = v_ptr + (offs_nextd[None, :] * stride_vd + offs_n[:, None] * stride_vn) + begin * stride_vn
        v = tl.load(
            v_ptrs,
            mask=((offs_nextd[None, :] < D) & (offs_n[:, None] < seqlen)),
            other=0.0,
        )
        p /= max_seq_len
        if use_mask:
            attn_mask = offs_m[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p = tl.where(attn_mask > 0, p, 0.0)
        p = p.to(v_ptr.dtype.element_ty)
        acc_j = tl.dot(p, v, allow_tf32=allow_tf32)
        acc += acc_j
        m_i = m_ij
    lse_i = m_i + tl.math.log(l_i)
    lse_i_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    lse_i_ptrs = lse_i_ptr + lse_i_offsets + begin
    tl.store(lse_i_ptrs, lse_i, mask=lse_i_offsets < seqlen)
    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_nextd[None, :] * stride_od + begin * stride_om)
    o_mask = (offs_m[:, None] < seqlen) & (offs_nextd[None, :] < D)
    tl.store(o_ptrs, acc, mask=o_mask)
