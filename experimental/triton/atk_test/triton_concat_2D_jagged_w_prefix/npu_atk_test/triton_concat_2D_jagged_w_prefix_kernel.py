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
# triton_concat_2D_jagged_w_prefix_kernel.py
# pylint: disable=chained-comparison

import torch

# torch 26.04 circular-import workaround for wrap_triton — do not remove
try:
    import torch._inductor.dependencies  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # 独立 ATK 目录不一定包含 Inductor 的可选依赖；目标 kernel 不依赖该模块。
    pass

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# concat_2D_jagged_w_prefix：值级融合版本（方案 1）
#
# 背景：NPU Ascend Triton 编译器不兼容"一个 program 内两个不同张量指针被选择/
# 合并"的结构——
#   * if/else 分支分别取 ValuesA/ValuesB 指针 → 分支汇合处 SSA phi 被误优化，
#     B 段 token 被静默读成错误源（case 1239：177 行 B 段全错）；
#   * tl.where 直接选指针 → OffsetAnalysis.cpp 断言失败，编译报错。
#
# 方案 1（本文件）：把"选哪个源"从指针层下沉到值层——
#   1. A/B 两条路径的指针都【无条件】计算，不产生任何分支；
#   2. 用掩码把不需要的那条 load 整体 predicate 掉（OOB 指针不触内存）；
#   3. 最后对【加载出的值】做 tl.where 选择（不选指针）。
# 结果：kernel 内无指针分支、无指针 tl.where，理论上绕过上述两个已证实缺陷。
#
# 函数签名与原版完全一致，API（triton_concat_2D_jagged_w_prefix_api.py）零改动。
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def concat_2D_jagged_w_prefix(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    n_prefix_from_B,  # nonzero is not supported when IS_REPLACE=True
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b

    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return

    offs_d = tl.arange(0, BLOCK_D)
    if IS_REPLACE:
        out_seq_start = seq_start_a + off_n
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        out_seq_start = seq_start_a + seq_start_b + off_n
        out_seq_b_start = seq_len_a + n_prefix_from_B

    out_ptrs = Out + out_seq_start.to(tl.int64) * stride_od + offs_d

    # ── 值级融合：选源条件只用于掩码 + 对【值】做 tl.where，不用于指针分支 ──
    use_a = (off_n < out_seq_b_start) & (off_n >= n_prefix_from_B)

    # A 侧行号（use_a=False 时该 load 被整体 masked off，越界/负指针不触内存）
    off_a = off_n - n_prefix_from_B
    if IS_DENSE_A:
        a_ptrs = ValuesA + off_a.to(tl.int64) * stride_ad + off_z.to(tl.int64) * stride_dense_batch + offs_d
    else:
        a_ptrs = ValuesA + (off_a + seq_start_a).to(tl.int64) * stride_ad + offs_d

    # B 侧行号（前缀修正用纯整数 select，不涉及指针选择）
    off_b = off_n - out_seq_b_start + n_prefix_from_B
    off_b += tl.where(off_n < n_prefix_from_B, out_seq_b_start - n_prefix_from_B, 0)
    if IS_DENSE_B:
        b_ptrs = ValuesB + off_b.to(tl.int64) * stride_bd + off_z.to(tl.int64) * stride_dense_batch + offs_d
    else:
        b_ptrs = ValuesB + (off_b + seq_start_b).to(tl.int64) * stride_bd + offs_d

    av = tl.load(a_ptrs, mask=use_a & (offs_d < D), other=0.0)
    bv = tl.load(b_ptrs, mask=(~use_a) & (offs_d < D), other=0.0)
    v = tl.where(use_a, av, bv)
    tl.store(out_ptrs, v, mask=offs_d < D)
