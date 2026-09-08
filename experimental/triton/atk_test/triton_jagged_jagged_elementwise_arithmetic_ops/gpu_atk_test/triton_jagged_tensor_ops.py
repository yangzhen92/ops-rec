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

from typing import Optional, Union

import torch
import triton  # @manual
import triton.language as tl  # @manual
from torch._tensor import Tensor


@triton.jit
def jagged_jagged_elementwise_arithmetic_ops(
    # pyre-fixme[2]: Parameter must be annotated.
    x_ptr,  # x_ptr and y_ptr is pointer of jagged tensor value
    # pyre-fixme[2]: Parameter must be annotated.
    y_ptr,
    M: tl.constexpr,  # M and N would be size of the tensor with (M , N)
    N: tl.constexpr,
    stride_row: tl.constexpr,  # shared row stride for tensor
    stride_col: tl.constexpr,  # shared colume stride for tensor
    # pyre-fixme[2]: Parameter must be annotated.
    output,
    thread_block_row_size: tl.constexpr,  # row and colume size of current thread block with size (thread_block_row_size * thread_block_col_size)
    thread_block_col_size: tl.constexpr,
    ops_func: tl.constexpr,  # function use for calculation either add or multiplication
) -> None:
    pid = tl.program_id(0)
    # number of col group need for total N col
    num_group_n = (N + thread_block_col_size - 1) // thread_block_col_size
    # pid position in col perspective in range(0,num_group_n)
    pid_n = pid % num_group_n
    # pid position in row perspective since everytime row increase when we have num_group_n iteration
    pid_m = pid // num_group_n

    offset_m = pid_m * thread_block_row_size + tl.arange(0, thread_block_row_size)
    offset_n = pid_n * thread_block_col_size + tl.arange(0, thread_block_col_size)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    offset = offset_m[:, None] * stride_row + offset_n[None, :] * stride_col

    x_ptr += offset
    y_ptr += offset

    x = tl.load(x_ptr, mask=mask)
    y = tl.load(y_ptr, mask=mask)

    if ops_func == "add":
        z = tensor_elementwise_add(x, y)
    else:
        z = tensor_elementwise_mul(x, y)

    output += offset
    tl.store(output, z, mask=mask)


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