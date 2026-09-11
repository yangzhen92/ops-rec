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
def jagged_dense_elementwise_mul_jagged_out_kernel(
    a_ptr,  # 1d jagged
    b_ptr,  # dense
    c_ptr,  # 1d jagged
    a_seq_lengths_ptr,
    a_offsets_ptr,
    stride_a,
    stride_bm,
    stride_bn,
    max_seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_row_block = tl.program_id(1)

    batch_offset = tl.load(a_offsets_ptr + pid_batch)
    batch_seq_len = tl.load(a_seq_lengths_ptr + pid_batch)
    truncated_seq_len = tl.minimum(batch_seq_len, max_seq_len)

    offs_row = tl.arange(0, BLOCK_M)
    offs_col = tl.arange(0, BLOCK_N)

    rows = pid_row_block * BLOCK_M + offs_row

    # a/c: start + batch offset + row offsets + initial col offsets；b: start + row/col offsets
    a_ptrs = a_ptr + batch_offset * stride_a + rows[:, None] * truncated_seq_len + offs_col[None, :]
    b_ptrs = b_ptr + rows[:, None] * stride_bm + offs_col[None, :] * stride_bn
    c_ptrs = c_ptr + batch_offset + rows[:, None] * truncated_seq_len + offs_col[None, :]

    for block_start in range(0, truncated_seq_len, BLOCK_N):
        cols = block_start + offs_col
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < truncated_seq_len) & (cols[None, :] < truncated_seq_len)
        a = tl.load(a_ptrs, mask=mask)
        a_ptrs += BLOCK_N

        b = tl.load(b_ptrs, mask=mask)
        b_ptrs += BLOCK_N

        c = a * b
        tl.store(c_ptrs, c, mask=mask)
        c_ptrs += BLOCK_N
