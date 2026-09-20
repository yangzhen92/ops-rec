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
def triton_jagged_to_dense_optimization_2d(
    input_jagged_values_ptr,
    input_jagged_offset_ptr,
    input_jagged_row_stride,
    output_dense_ptr,
    output_dense_row_stride,
    output_dense_matrix_stride,
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    padded_value,
    operation_function: tl.constexpr,
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    offset_idx = pid

    begin = tl.load(input_jagged_offset_ptr + offset_idx)
    end = tl.load(input_jagged_offset_ptr + offset_idx + 1)

    cur_jagged_tensor_row_size = end - begin

    output_dense_ptr += pid * output_dense_matrix_stride
    input_jagged_values_ptr += begin * input_jagged_row_stride

    if operation_function is not None:
        operation_dense += pid * output_dense_matrix_stride

    offset_row = tl.arange(0, thread_block_row_size)

    dense_col_size = output_dense_row_stride
    dense_row_size = output_dense_matrix_stride // output_dense_row_stride

    for _i in range(0, dense_row_size, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = offset_row[:, None] * output_dense_row_stride + offset_col[None, :]

        for _j in range(0, dense_col_size, thread_block_col_size):
            dense_mask = (offset_row[:, None] < dense_row_size) & (offset_col[None, :] < dense_col_size)
            jagged_mask = (offset_row[:, None] < cur_jagged_tensor_row_size) & (
                offset_col[None, :] < input_jagged_row_stride
            )

            jagged_val = tl.load(
                input_jagged_values_ptr + block_offset,
                mask=jagged_mask,
                other=padded_value,
            )

            if operation_function is not None:
                operation_dense_val = tl.load(operation_dense + block_offset, mask=dense_mask, other=0.0)
                jagged_val = operation_function(operation_dense_val, jagged_val)

            tl.store(output_dense_ptr + block_offset, jagged_val, mask=dense_mask)

            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size
