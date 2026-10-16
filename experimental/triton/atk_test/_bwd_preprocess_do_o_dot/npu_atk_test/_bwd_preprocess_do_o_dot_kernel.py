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
        triton.Config({"BLOCK_T": 8, "multibuffer": True}),
        triton.Config({"BLOCK_T": 16, "multibuffer": True}),
        triton.Config({"BLOCK_T": 32, "multibuffer": True}),
        triton.Config({"BLOCK_T": 64, "multibuffer": True}),
        triton.Config({"BLOCK_T": 128, "multibuffer": True}),
    ],
    key=["T"],
)
@triton.jit
def _bwd_preprocess_do_o_dot(
    o_ptr,  # 前向输出（输入）
    do_ptr,  # 输出梯度（输入）
    delta_ptr,  # 计算结果（输出）
    T,  # 序列长度（时间步数）
    stride_ob,  # o 在 batch 维度上的步长
    stride_ot,  # o 在 T 维度上的步长
    stride_od,  # o 在 D 维度上的步长
    stride_do_b,  # do 在 batch 维度上的步长
    stride_do_t,  # do 在 T 维度上的步长
    stride_do_d,  # do 在 D 维度上的步长
    BLOCK_T: tl.constexpr,  # T 维度的块大小
    BLOCK_D: tl.constexpr,  # D 维度的块大小
):
    start_t = tl.program_id(0)
    offs_t = start_t * BLOCK_T + tl.arange(0, BLOCK_T)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob  # batch 偏移
        + offs_t[:, None] * stride_ot  # T 维度偏移（广播到行）
        + offs_d[None, :] * stride_od  # D 维度偏移（广播到列）
    )
    do_ptrs = do_ptr + pid_b * stride_do_b + offs_t[:, None] * stride_do_t + offs_d[None, :] * stride_do_d

    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))
