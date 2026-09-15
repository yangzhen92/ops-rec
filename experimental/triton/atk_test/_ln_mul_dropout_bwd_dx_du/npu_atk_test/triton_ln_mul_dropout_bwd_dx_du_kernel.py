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


import torch

# torch 26.04 circular-import workaround for wrap_triton — do not remove
import torch._inductor.dependencies  # noqa: F401

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}),
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
        triton.Config({"BLOCK_D": 512}),
        triton.Config({"BLOCK_D": 1024}),
        triton.Config({"BLOCK_D": 2048}),
        triton.Config({"BLOCK_D": 4096}),
    ],
    key=["D"],
)
@triton.jit
def _ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du0,
    stride_du1,  # do we need this?
    stride_dy,
    stride_x,
    stride_u0,
    stride_u1,
    stride_y,
    D,
    eps,
    seed,
    dropout_ratio,
    N,
    NUM_U_SLICES: tl.constexpr,
    UD: tl.constexpr,  # valid only when NUM_U_SLICES>1
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u0
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du0
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    for idx in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            random_offsets = row * BLOCK_D + cols
            if CONCAT_UX:
                # apply dropout on du
                random_du = tl.rand(seed, random_offsets)
                du_keep = (random_du > dropout_ratio) | (dropout_ratio == 0.0)
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dx
                random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = (random_dx > dropout_ratio) | (dropout_ratio == 0.0)
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = (random_dy > dropout_ratio) | (dropout_ratio == 0.0)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = (random > dropout_ratio) | (dropout_ratio == 0.0)
                # write-back
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln

        # 2D write-back du
        if NUM_U_SLICES > 1:
            du = du.reshape(NUM_U_SLICES, UD)
            u_tile_offsets = tl.arange(0, NUM_U_SLICES)
            u_inner_offsets = tl.arange(0, UD)
            offsets_2d = u_tile_offsets[:, None] * stride_du1 + u_inner_offsets[None, :]
            u_mask = (u_tile_offsets[:, None] < NUM_U_SLICES) & (u_inner_offsets[None, :] < UD)
            tl.store(DU + offsets_2d, du, mask=u_mask)
        else:
            tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)

        # load u
        if NUM_U_SLICES > 1:
            u_tile_offsets = tl.arange(0, NUM_U_SLICES)
            u_inner_offsets = tl.arange(0, UD)
            offsets_2d = u_tile_offsets[:, None] * stride_u1 + u_inner_offsets[None, :]
            u_mask = (u_tile_offsets[:, None] < NUM_U_SLICES) & (u_inner_offsets[None, :] < UD)
            u = tl.load(U + offsets_2d, mask=u_mask, other=0.0).to(tl.float32)
            u = u.reshape(NUM_U_SLICES * UD)
        else:
            u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)

        dy = dy * u
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                if CONCAT_UX:
                    u = tl.where(
                        du_keep,  # pyre-ignore [61]
                        u / (1.0 - dropout_ratio),
                        0.0,
                    )
                    x = tl.where(
                        dx_keep,  # pyre-ignore [61]
                        x / (1.0 - dropout_ratio),
                        0.0,
                    )
                    y = tl.where(
                        dy_keep,  # pyre-ignore [61]
                        y / (1.0 - dropout_ratio),
                        0.0,
                    )
                else:
                    y = tl.where(
                        dy_keep,  # pyre-ignore [61]
                        y / (1.0 - dropout_ratio),
                        0.0,
                    )
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw = dy * xhat
        partial_db = dy
        # First store doesn't accumulate
        if idx > 0:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u0
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du0
        row += tile_num
