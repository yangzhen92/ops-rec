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


# Similar to fwd kernel, this one is using a grid of
# (num_blocks_m, B) where num_blocks_m is seq_len / BLOCK_SIZE_M
@triton.jit
def _jagged_flash_attention_bwd_preprocess_basic_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_od = tl.arange(0, BLOCK_SIZE_D)

    o_offsets = offs_om[:, None] * stride_om + offs_od[None, :] * stride_od + begin_o * stride_om
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load O
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(delta_ptr + begin_o + offs_om, delta, mask=offs_om < M)
