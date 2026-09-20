#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import torch

# torch 26.04 circular-import workaround for wrap_triton — do not remove
import torch._inductor.dependencies  # noqa: F401

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────────────────
# _layer_norm_bwd_dx 的 autotune 配置
#
# 写法参考 double_benchmark/examples/triton_jagged_to_dense.py:
# 扁平的内联 configs 列表；NPU 上无需对 num_warps 做 autotune。
# BLOCK_D 决定每个 program（一行）覆盖的特征维 width。必须 >= next_pow2(D)，
# 否则 kernel 只处理每行的前 BLOCK_D 列（结果错误）；同时受 64KB/元素 限制。
# ────────────────────────────────────────────────────────────────────────────
def _layer_norm_bwd_dx_prune_configs(configs, named_args, **kwargs):
    """按 D 与 dtype 剪枝无效配置。

    - BLOCK_D 必须覆盖整行：BLOCK_D >= next_pow2(D)，否则输出会缺列；
    - BLOCK_D 受 64KB/元素 限制：BLOCK_D <= 65536 // element_size()。
    """
    d = named_args.get("D", 0)
    x = named_args.get("X")
    element_size = 1
    if x is not None:
        try:
            element_size = x.element_size()
        except Exception:
            element_size = 1
    max_fused = 65536 // element_size
    min_block = triton.next_power_of_2(d)
    kept = []
    for config in configs:
        block_d = config.kwargs["BLOCK_D"]
        if min_block <= block_d <= max_fused:
            kept.append(config)
    return kept


# each CTA calculates one row of dx
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
        triton.Config({"BLOCK_D": 512}),
        triton.Config({"BLOCK_D": 1024}),
        triton.Config({"BLOCK_D": 2048}),
        triton.Config({"BLOCK_D": 4096}),
        triton.Config({"BLOCK_D": 8192}),
        triton.Config({"BLOCK_D": 16384}),
        triton.Config({"BLOCK_D": 32768}),
        triton.Config({"BLOCK_D": 65536}),
    ],
    key=["D"],
    prune_configs_by={"early_config_prune": _layer_norm_bwd_dx_prune_configs},
)
@triton.jit
def _layer_norm_bwd_dx(
    DX,
    DY,
    X,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)
