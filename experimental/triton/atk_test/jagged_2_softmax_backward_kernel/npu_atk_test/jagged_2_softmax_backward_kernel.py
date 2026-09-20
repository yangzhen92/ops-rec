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
# pyre-unsafe

import triton
import triton.language as tl


@triton.jit
def jagged_2_softmax_backward_kernel(
    grad_output_ptr,  # input
    softmax_output_ptr,
    grad_input_ptr,  # return value
    offsets_row_ptr,
    offsets_col_ptr,
    offsets_overall_ptr,
    grad_output_stride,
    softmax_output_stride,
    grad_input_stride,
    transpose,  # transpose
    max_seq_len_row: tl.constexpr,
    max_seq_len_col: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    begin = tl.load(offsets_overall_ptr + pid_batch)
    if transpose:
        N = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(offsets_row_ptr + pid_batch)
        H = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(offsets_col_ptr + pid_batch)
        stride_n = H
        stride_h = H // H  # 1
        H = tl.minimum(max_seq_len_col, H)
        N = tl.minimum(max_seq_len_row, N)
    else:
        N = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(offsets_col_ptr + pid_batch)
        H = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(offsets_row_ptr + pid_batch)
        stride_h = N
        stride_n = N // N  # 1
        H = tl.minimum(max_seq_len_row, H)
        N = tl.minimum(max_seq_len_col, N)
    if pid_head >= H:
        return
    if H == 0 or N == 0:
        pass
    start_ptr = grad_output_ptr + begin * grad_output_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = start_ptr + offsets * grad_output_stride * stride_n + pid_head * grad_output_stride * stride_h
    softmax_output_ptrs = (
        softmax_output_ptr
        + begin * softmax_output_stride
        + offsets * softmax_output_stride * stride_n
        + pid_head * softmax_output_stride * stride_h
    )
    grad_output_row = tl.load(grad_output_ptrs, mask=offsets < N, other=0.0)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=offsets < N, other=0.0)
    sum_value = tl.sum(grad_output_row * softmax_output_row, axis=0)
    grad_input_row = (grad_output_row - sum_value) * softmax_output_row
    grad_input_row_start_ptr = grad_input_ptr + begin * grad_input_stride
    grad_input_ptrs = (
        grad_input_row_start_ptr + offsets * grad_input_stride * stride_n + pid_head * grad_input_stride * stride_h
    )
    tl.store(grad_input_ptrs, grad_input_row, mask=offsets < N)
