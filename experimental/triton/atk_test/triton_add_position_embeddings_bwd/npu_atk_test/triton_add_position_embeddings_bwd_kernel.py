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
def _add_position_embeddings_bwd_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    DenseOut,
    JaggedOut,
    B,
    D,
    scale,
    rows_per_core,
    stride_jn,
    stride_jon,
    stride_don,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_d = tl.arange(0, BLOCK_D)
    core_id = tl.program_id(0)
    off_k_base = core_id * rows_per_core

    for i in tl.static_range(0, BLOCK_K):
        off_k = off_k_base + i
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for off_b in range(0, B):
            max_ind = tl.load(high_inds + off_b)
            if off_k < max_ind:
                seq_start = tl.load(seq_offsets + off_b)
                jagged_ptr = (
                    Jagged
                    + seq_start * stride_jn
                    + off_k * stride_jn
                )
                jagged_ptrs = jagged_ptr + offs_d
                jg = tl.load(
                    jagged_ptrs,
                    mask=offs_d < D,
                )
                accumulator += jg
                if SCALE_JAGGED:
                    out_jagged_ptr = (
                        JaggedOut
                        + seq_start * stride_jon
                        + off_k * stride_jon
                    )
                    out_jagged_ptrs = out_jagged_ptr + offs_d
                    tl.store(
                        out_jagged_ptrs,
                        jg * scale,
                        mask=offs_d < D,
                    )
            elif off_k == max_ind:
                seq_start = tl.load(seq_offsets + off_b)
                seq_end = tl.load(seq_offsets + off_b + 1)
                for k in range(seq_start + max_ind, seq_end):
                    jagged_ptr = Jagged + k * stride_jn
                    jagged_ptrs = jagged_ptr + offs_d
                    jg = tl.load(
                        jagged_ptrs,
                        mask=offs_d < D,
                    )
                    accumulator += jg
                    if SCALE_JAGGED:
                        out_jagged_ptr = JaggedOut + k * stride_jon
                        out_jagged_ptrs = out_jagged_ptr + offs_d
                        tl.store(
                            out_jagged_ptrs,
                            jg * scale,
                            mask=offs_d < D,
                        )
        out = accumulator.to(DenseOut.dtype.element_ty)
        out_ptrs = DenseOut + off_k * stride_don + offs_d
        tl.store(
            out_ptrs,
            out,
            mask=offs_d < D,
        )