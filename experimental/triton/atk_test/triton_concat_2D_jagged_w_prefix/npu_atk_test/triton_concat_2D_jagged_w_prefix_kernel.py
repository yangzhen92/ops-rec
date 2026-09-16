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

import torch

try:
    import torch._inductor.dependencies  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pass

import triton
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
    ],
    key=["D"],
)
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
    n_prefix_from_B,
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

    if IS_REPLACE:
        out_seq_start = seq_start_a + off_n
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        out_seq_start = seq_start_a + seq_start_b + off_n
        out_seq_b_start = seq_len_a + n_prefix_from_B

    out_base = Out + out_seq_start.to(tl.int64) * stride_od
    if off_n < out_seq_b_start and off_n >= n_prefix_from_B:
        off_a = off_n - n_prefix_from_B
        if IS_DENSE_A:
            in_base = (
                ValuesA
                + off_a.to(tl.int64) * stride_ad
                + off_z.to(tl.int64) * stride_dense_batch
            )
        else:
            in_base = ValuesA + (off_a + seq_start_a).to(tl.int64) * stride_ad
    else:
        off_b = off_n - out_seq_b_start + n_prefix_from_B
        if off_n < n_prefix_from_B:
            off_b += out_seq_b_start - n_prefix_from_B
        if IS_DENSE_B:
            in_base = (
                ValuesB
                + off_b.to(tl.int64) * stride_bd
                + off_z.to(tl.int64) * stride_dense_batch
            )
        else:
            in_base = ValuesB + (off_b + seq_start_b).to(tl.int64) * stride_bd

    for chunk in tl.range(0, D, BLOCK_D):
        cols = chunk + tl.arange(0, BLOCK_D)
        mask = cols < D
        value = tl.load(in_base + cols, mask=mask)
        tl.store(out_base + cols, value, mask=mask)
