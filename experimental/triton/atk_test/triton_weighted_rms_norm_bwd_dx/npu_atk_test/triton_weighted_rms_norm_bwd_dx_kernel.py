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
# triton_weighted_rms_norm_bwd_dx_kernel.py
# pylint: disable=chained-comparison
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

#!/usr/bin/env python3


import torch

# torch 26.04 circular-import workaround for wrap_triton — do not remove
import torch._inductor.dependencies  # noqa: F401

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

# ── _weighted_rms_norm_bwd_dx autotune 配置 ─────────────────────────────
# 每 program 处理一行（grid=(N,)），BLOCK_D 必须 >= next_pow2(D) 且 <= _AUTOTUNE_MAX_BLOCK_D(8192, NPU UB 硬上限)
# 。
_AUTOTUNE_MAX_BLOCK_D = 8192  # NPU UB 硬上限（见 API _MAX_BLOCK_D）


def _weighted_rms_norm_bwd_dx_prune_configs(configs, named_args, **kwargs):
    """按 D 剪枝无效配置。

    - BLOCK_D 必须覆盖整行：BLOCK_D >= next_pow2(D)，否则输出缺列；
    - BLOCK_D 受 NPU UB 限制：BLOCK_D <= _AUTOTUNE_MAX_BLOCK_D（8192），
      超过会 ub overflow 编译失败。
    """
    d = kwargs.get("D", 0)
    min_block = triton.next_power_of_2(d)
    kept = []
    for config in configs:
        block_d = config.kwargs["BLOCK_D"]
        if min_block <= block_d <= _AUTOTUNE_MAX_BLOCK_D:
            kept.append(config)
    return kept


# each CTA calculates one row of _weighted_rms_norm_bwd_dx
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
    ],
    key=["D"],
    prune_configs_by={"early_config_prune": _weighted_rms_norm_bwd_dx_prune_configs},
)
@triton.jit
def _weighted_rms_norm_bwd_dx(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    Lock,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    GROUP_N,
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
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    wdy = w * dy

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - (xhat * c1)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    DW = DW + lock_id * D + cols
    # Accumulate partial sums for dw/db
    # NPU 兼容写法：直接 atomic_add 到 DW（需 DW 初始化为 0）。
    # 原 CUDA 版本用 Lock 自旋锁 + Count 判断首个 program 直接存、后续累加；
    # 数学上等价于每个 program 都把 partial_dw 累加到初始为 0 的 DW，
    # 因此 DX 之外的后处理（_rms_norm_bwd_dwdb 对 DW 逐组求和）仍能拿到正确权重梯度。
    # NPU BiSheng 编译器不支持 while(tl.atomic_cas()) 自旋锁，会报
    # 'hivm.hir.load' op using value defined outside the regio
    partial_dw = (dy * xhat).to(tl.float32)
    tl.atomic_add(DW, partial_dw, mask=mask)
