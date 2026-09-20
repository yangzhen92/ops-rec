#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
# triton_weighted_rms_norm_fwd_kernel.py

# Work around a circular import in torch 26.04 wrap_triton.
import torch._inductor.dependencies  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _weighted_rms_norm_fwd(
    X,
    Y,
    W,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = y * w
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
