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

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4, "multibuffer": True}),
        triton.Config({"BLOCK_H": 8, "multibuffer": True}),
        triton.Config({"BLOCK_H": 16, "multibuffer": True}),
        triton.Config({"BLOCK_H": 32, "multibuffer": True}),
        triton.Config({"BLOCK_H": 64, "multibuffer": True}),
        triton.Config({"BLOCK_H": 128, "multibuffer": True}),
    ],
    key=["Heads", "D"],
)
@triton.jit
def _group_norm_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u0,
    stride_y,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):  # type: ignore
    row = tl.program_id(0)  # batch 分块
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u0
    Y += row.to(tl.int64) * stride_y
    start_h = tl.program_id(1)  # Head 分块
    cols = tl.arange(0, BLOCK_D)
    heads = start_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets = heads[:, None] * D + cols[None, :]
    mask_h = heads < Heads
    mask_c = cols < D
    mask = mask_c[None, :] & mask_h[:, None]

    # Compute mean
    mean = 0.0
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    mean = tl.ravel(mean)

    # Compute variance
    _var = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row * Heads + heads, mean, mask=mask_h)
    tl.store(Rstd + row * Heads + heads, rstd, mask=mask_h)

    # Normalize and apply linear transformation
    y = x_mean * rstd[:, None]  # pyre-ignore [16]
    w = tl.load(W + heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + heads, mask=mask_h).to(tl.float32)
    y = y * w[:, None] + b[:, None]
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    y = y * u

    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = (random_u > dropout_ratio) | (dropout_ratio == 0.0)
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + Heads * D)
            x_keep = (random_x > dropout_ratio) | (dropout_ratio == 0.0)
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * Heads * D)
            y_keep = (random_y > dropout_ratio) | (dropout_ratio == 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            y_keep = (random > dropout_ratio) | (dropout_ratio == 0.0)
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
