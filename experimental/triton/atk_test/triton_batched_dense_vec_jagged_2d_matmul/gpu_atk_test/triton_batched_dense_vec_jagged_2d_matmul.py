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
def triton_batched_dense_vec_jagged_2d_matmul(
    jagged_tensor_ptr,
    dense_ptr,
    jagged_offset,
    thread_block_col_size: tl.constexpr,
    dense_row_stride,
    jagged_value_row_stride,
    D,
    H: tl.constexpr,
    output_ptr,
) -> None:
    pid = tl.program_id(0)

    # number of kernel need for with matrix (N,D) calculated by D // thread_block_col_size
    GRID_DIM_COL = (D + thread_block_col_size - 1) // thread_block_col_size

    # current output row index
    output_row_idx = pid // GRID_DIM_COL

    # current jagged tensor offset index
    jagged_offset_id = output_row_idx // H

    # current index with D reference since the real shape of jagged values is [B , N , H * D]
    D_refer_idx = output_row_idx % H

    # current part of [N * D] id
    group_id = pid % GRID_DIM_COL

    # size of tile
    offset = group_id * thread_block_col_size + tl.arange(0, thread_block_col_size)

    # begin index and end index of values
    begin = tl.load(jagged_offset + jagged_offset_id)
    end = tl.load(jagged_offset + (jagged_offset_id + 1))

    # update each pointer to the correct address
    dense_ptr += output_row_idx * dense_row_stride
    jagged_tensor_ptr += begin * jagged_value_row_stride + D_refer_idx * D
    output_ptr += D * output_row_idx

    # Number of row each kernel will go through
    num_row = tl.minimum(end - begin, dense_row_stride)

    # accumulation variable use for matmul
    mask = offset < D
    val1_init = tl.load(dense_ptr)
    val2_init = tl.load(jagged_tensor_ptr + offset, mask=mask, other=0.0)
    acc_dtype = (val1_init * val2_init).dtype
    acc = tl.zeros((thread_block_col_size,), dtype=acc_dtype)

    for i in range(num_row):
        val1 = tl.load(dense_ptr + i)
        val2 = tl.load(jagged_tensor_ptr + offset, mask=mask, other=tl.cast(0.0, tl.float32))
        result = val1 * val2
        acc += result
        jagged_tensor_ptr += jagged_value_row_stride

    tl.store(output_ptr + offset, acc, mask=mask)
