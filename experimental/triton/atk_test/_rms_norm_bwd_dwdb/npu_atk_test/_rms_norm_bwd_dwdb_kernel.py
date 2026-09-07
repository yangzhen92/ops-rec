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


# ============ autotune 版本 ============
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 32, "BLOCK_D": 128, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 32, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 128, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "BLOCK_D": 256, "multibuffer": True}),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 128, "BLOCK_D": 128, "multibuffer": True}),
        triton.Config({"BLOCK_N": 256, "BLOCK_D": 64, "multibuffer": True}),
    ],
    key=["N", "D"],
)
@triton.jit
def _rms_norm_bwd_dwdb_kernel(
    DW,
    FINAL_DW,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
