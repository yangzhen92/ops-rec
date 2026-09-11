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
# each kernel will handle the conversion of one jagged tensor offset range from corresponding dense index
@triton.jit
def triton_dense_to_jagged(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    jagged_value_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    dense_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic opeartion function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_jagged_value_ptr,
) -> None:
    pid = tl.program_id(0)

    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # size of the current value offset range (M , N)
    N = jagged_value_row_stride
    M = end - begin

    dense_boundary_col = dense_row_stride
    # tl.minimum will change the return type cased compile issue
    # in that case use if statement instead
    if N < dense_row_stride:
        dense_boundary_col = N

    dense_boundary_row = tl.minimum(dense_matrix_stride // dense_row_stride, M)

    jagged_value_ptr += begin * jagged_value_row_stride
    if JAGGED_DIM > 2:
        dense_indice = tl.load(dense_indices_ptr + pid)
        # if dense output range we set dense_boundary to -1
        # that mean dense values will not be use with mask
        # since we still need the calculation of fusion step
        # therefore we do not do return here
        if dense_indice == -1:
            dense_boundary_col = -1
        else:
            output_dense_ptr += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

    if operation_function is not None:
        operation_jagged_value_ptr += begin * jagged_value_row_stride

    offset_row = tl.arange(0, thread_block_row_size)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = offset_row[:, None] * dense_row_stride + offset_col[None, :] * dense_col_stride

        for _j in range(0, N, thread_block_col_size):
            dense_mask = (offset_row[:, None] < dense_boundary_row) & (offset_col[None, :] < dense_boundary_col)
            jagged_mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            dense_values = tl.load(output_dense_ptr + block_offset, mask=dense_mask, other=0)
            if operation_function is not None:
                operation_jagged_value = tl.load(operation_jagged_value_ptr + block_offset, mask=jagged_mask, other=0)
                if operation_function == "add":
                    dense_values = tensor_elementwise_add(dense_values, operation_jagged_value)
                else:
                    dense_values = tensor_elementwise_mul(dense_values, operation_jagged_value)
            tl.store(jagged_value_ptr + block_offset, dense_values, mask=jagged_mask)
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size


# this function parse the jagged tensor offsets to corresponding dense index position
# to see the detail of it see the quip note : https://fb.quip.com/gnzpA7d13vqO
# the FBGEMM implementation refer : https://www.internalfb.com/code/fbsource/[308212b2902c3182edcb5b204768321e032e8175]/fbcode/deeplearning/fbgemm/fbgemm_gpu/src/jagged_tensor_ops.cu?lines=280
# In FBGEMM it was computed by GPU but in triton currently has some compilation issue so we use CUP computation method as workaround
# However in real-world case if we only dealing with 2d jagged tensor we don't need to use this function at all
def _jagged_offsets_to_dense_indice(
    offsets: list[torch.Tensor], dense_strides: list[int], dense_sizes: list[int]
) -> torch.Tensor:
    output_offset = torch.zeros(len(offsets[-1]) - 1, device="cpu", dtype=torch.int32)

    offsets_cpu = []

    for offset in offsets:
        offsets_cpu.append(offset.cpu())

    for i in range(0, len(offsets_cpu[-1]) - 1):
        idx = i
        result = 0

        # flag to check if current offset is in the range of dense
        in_range = True
        for j in range(len(offsets_cpu) - 2, -1, -1):
            left = 0
            right = offsets_cpu[j].size(0)

            # binary search found the corresponding offset group of current index
            while left < right:
                mid = left + (right - left) // 2

                if offsets_cpu[j][mid] > idx:
                    right = mid
                else:
                    left = mid + 1

            cur_val = idx - offsets_cpu[j][left - 1]

            if dense_sizes and cur_val >= dense_sizes[j + 1]:
                in_range = False
                break

            result += cur_val * dense_strides[j + 1]
            idx = left - 1

        if in_range:
            result += idx * dense_strides[0]

            # another out of output dense range case
            if dense_sizes and idx > dense_sizes[0]:
                result = -1
            output_offset[i] = result
        else:
            output_offset[i] = -1

    return output_offset
