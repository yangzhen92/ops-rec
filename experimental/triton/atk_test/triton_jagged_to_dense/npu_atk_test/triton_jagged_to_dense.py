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

# pyre-strict

# pyre-ignore-all-errors[6]

import torch
import triton  # @manual
import triton.language as tl  # @manual

@triton.jit
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def tensor_elementwise_add(x, y):
    return x + y


@triton.jit
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def tensor_elementwise_mul(x, y):
    return x * y

@triton.autotune(
    configs=[
        triton.Config({"thread_block_row_size": 16, "thread_block_col_size": 16}),
        triton.Config({"thread_block_row_size": 32, "thread_block_col_size": 32}),
        triton.Config({"thread_block_row_size": 64, "thread_block_col_size": 64}),
        triton.Config({"thread_block_row_size": 128, "thread_block_col_size": 128}),
        triton.Config({"thread_block_row_size": 256, "thread_block_col_size": 256}),
    ],
    key=["JAGGED_DIM"],
)
@triton.jit
def triton_jagged_to_dense(
    # only constexpr annotations support in triton now
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    # pyre-fixme[2]: Parameter must be annotated.
    dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic operation function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # begin index and end index of jagged tensor Values
    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # adjust the address of the jagged tensor Values to the correct address
    jagged_value_ptr += begin * jagged_value_row_stride

    # if it's 2D (or 1D) Jagged tensor we can direct use the offset in offsets ( since there is only one offset )
    # else we actually need to use the preprocess index to found the correct address of dense
    if JAGGED_DIM > 2:
        # read the index for current kernel
        dense_indice = tl.load(dense_indices_ptr + pid)

        # if the dense_indice is -1 which mean it's a truncation case
        # in that case we don't need to do anything since the dense
        # initialize with padded value
        if dense_indice == -1:
            return

        # adjust the address of output dense ptr to the correct address
        output_dense_ptr += dense_indice

        # also need to update the operation function if exist
        # notice dense_indice of two is same because we assume
        # the two dense + dense are same size
        if operation_function is not None:
            operation_dense += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

        if operation_function is not None:
            operation_dense += pid * dense_matrix_stride

    offset_row = tl.arange(0, thread_block_row_size)

    # boundary need for the mask since it could be dense's size smaller than jagged tensor or revert case
    N = tl.minimum(dense_row_stride, jagged_value_row_stride)
    M = tl.minimum(dense_matrix_stride // dense_row_stride, end - begin)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )
        for _j in range(0, N, thread_block_col_size):
            mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            jagged_val = tl.load(jagged_value_ptr + block_offset, mask=mask, other=0)

            # if there is some arithmetic operation we do the fusion computation
            if operation_function is not None:
                val1 = jagged_val
                val2 = tl.load(operation_dense + block_offset, mask=mask, other=0)
                # do the arithmetic operation
                if operation_function == "add":
                    jagged_val = tensor_elementwise_add(val1, val2)
                else:
                    jagged_val = tensor_elementwise_mul(val1, val2)

            # store the result
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=mask)

            # update the block offset
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size
