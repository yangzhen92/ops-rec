# Copyright (c) Meta Platforms, Inc. and affiliates.
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
# _layer_norm_fwd 的 autotune 配置
# BLOCK_D 决定每个 program（一行）覆盖的特征维 width。必须 >= next_pow2(D)，
# 否则 kernel 只处理每行的前 BLOCK_D 列（结果错误）；同时受 64KB/元素 限制。
# ────────────────────────────────────────────────────────────────────────────
def _layer_norm_fwd_prune_configs(configs, named_args, **kwargs):
    """按 D 与 dtype 剪枝无效配置。

    - BLOCK_D 必须覆盖整行：BLOCK_D >= next_pow2(D)，否则结果错误；
    - BLOCK_D 受 64KB/元素 限制：BLOCK_D <= 65536 // element_size()。
    """
    d = kwargs.get("D", 0)
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
    prune_configs_by={"early_config_prune": _layer_norm_fwd_prune_configs},
)
@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x, axis=0) / D
    else:
        mean = tl.load(Mean + row)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = tl.zeros([BLOCK_D], dtype=tl.float32)
        _var += x_mean * x_mean
        var = tl.sum(_var, axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
    else:
        rstd = tl.load(Rstd + row)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
