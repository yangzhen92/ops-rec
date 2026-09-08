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


from typing import Optional, Tuple

import torch

# # torch 26.04 circular-import workaround for wrap_triton — do not remove
# import torch._inductor.dependencies  # noqa: F401

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from common import (
    switch_to_contiguous_if_needed,
)


@triton.jit
def _ln_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u0,
    stride_u1,
    stride_y,
    NUM_U_SLICES: tl.constexpr,  # num of heads
    UD: tl.constexpr,  # valid only when NUM_U_SLICES>1
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u0  # next rows
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    # Compute mean
    mean = 0.0
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = y * w + b

    if NUM_U_SLICES > 1:
        u_tile_offsets = tl.arange(0, NUM_U_SLICES)
        u_inner_offsets = tl.arange(0, UD)
        offsets_2d = u_tile_offsets[:, None] * stride_u1 + u_inner_offsets[None, :]
        u_mask = (u_tile_offsets[:, None] < NUM_U_SLICES) & (u_inner_offsets[None, :] < UD)
        u = tl.load(U + offsets_2d, mask=u_mask, other=0.0).to(tl.float32)
        u = u.reshape(NUM_U_SLICES * UD)
    else:
        u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)

    # u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)
    y = y * u

    if TRAINING:
        random_offsets = row * BLOCK_D + cols
        if CONCAT_UX:
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = (random_u > dropout_ratio) | (dropout_ratio == 0.0)
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + D)
            x_keep = (random_x > dropout_ratio) | (dropout_ratio == 0.0)
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * D)
            y_keep = (random_y > dropout_ratio) | (dropout_ratio == 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random = tl.rand(seed, random_offsets)
            y_keep = (random > dropout_ratio) | (dropout_ratio == 0.0)
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


def triton_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:  # y, mean, rstd, BLOCK_D, num_warps, seed
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D
    U_IS_3D = u.dim() == 3
    if concat_ux:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    # pyre-ignore[28]
    torch.library.wrap_triton(_ln_mul_dropout_fwd)[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        D,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        u.stride(1) if U_IS_3D else 1,
        y.stride(0),
        NUM_U_SLICES=u.size(1) if U_IS_3D else 1,
        UD=u.size(-1) if U_IS_3D else D,
        BLOCK_D=BLOCK_D,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        num_warps=num_warps,
    )
    return y, mean, rstd, BLOCK_D, num_warps, seed  # pyre-ignore [7]
