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
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "multibuffer": True}),
        triton.Config({"BLOCK_N": 32, "multibuffer": True}),
        triton.Config({"BLOCK_N": 64, "multibuffer": True}),
        triton.Config({"BLOCK_N": 128, "multibuffer": True}),
        triton.Config({"BLOCK_N": 256, "multibuffer": True}),
        triton.Config({"BLOCK_N": 512, "multibuffer": True}),
    ],
    key=["B", "D"],
)
@triton.jit
def _add_position_embeddings_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    Dense,
    Out,
    B,
    D,
    ncores,
    scale,
    ind_offsets,
    stride_jn,
    stride_dk,
    stride_on,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    rows_per_core = tl.cdiv(B, ncores)
    row_cur = pid * rows_per_core
    row_end = tl.minimum(row_cur + rows_per_core, B)

    if row_cur >= row_end:
        return

    # 找到 row_cur 所在的 batch
    batch_idx = 0
    next_batch_start = tl.load(seq_offsets + 1)
    while row_cur >= next_batch_start:
        batch_idx += 1
        next_batch_start = tl.load(seq_offsets + batch_idx + 1)

    # 因为 BLOCK_D = D，直接使用整个 D 维
    d_offs = tl.arange(0, BLOCK_D).to(tl.int32)
    d_mask = d_offs < D

    # 外层：按 batch 迭代
    while row_cur < row_end:
        batch_seq_start = tl.load(seq_offsets + batch_idx)
        batch_seq_end = tl.load(seq_offsets + batch_idx + 1)
        batch_max_ind = tl.load(high_inds + batch_idx)
        batch_ind_offset = None
        if ind_offsets is not None:
            batch_ind_offset = tl.load(ind_offsets + batch_idx)

        batch_end_in_core = tl.minimum(batch_seq_end, row_end)
        nrows = batch_end_in_core - row_cur
        base_pos = row_cur - batch_seq_start

        # 中层：BLOCK_N 行分块
        for local_start in range(0, nrows, BLOCK_N):
            tile_nrows = tl.minimum(BLOCK_N, nrows - local_start)

            for i in range(tile_nrows):
                row_idx = row_cur + local_start + i
                pos = base_pos + local_start + i

                clamped = tl.minimum(pos, batch_max_ind)
                if ind_offsets is not None:
                    clamped = tl.minimum(pos + batch_ind_offset, batch_max_ind)

                jg = tl.load(
                    Jagged + row_idx * stride_jn + d_offs,
                    mask=d_mask,
                )
                if SCALE_JAGGED:
                    jg = jg * scale

                dn = tl.load(
                    Dense + clamped * stride_dk + d_offs,
                    mask=d_mask,
                )
                tl.store(
                    Out + row_idx * stride_on + d_offs,
                    jg + dn,
                    mask=d_mask,
                )

        row_cur = batch_end_in_core
        batch_idx += 1


def add_position_embeddings(
    jagged: torch.Tensor,
    seq_offsets: torch.Tensor,
    high_inds: torch.Tensor,
    dense: torch.Tensor,
    scale: float = 1.0,
    ind_offsets: torch.Tensor = None,
    scale_jagged: bool = False,
) -> torch.Tensor:
    B = seq_offsets.numel() - 1
    D = jagged.shape[1]

    out = torch.empty_like(jagged)

    device_props = triton.runtime.driver.active.utils.get_device_properties(jagged.device)
    ncores = device_props.get("num_vectorcore", -1)
    if ncores is None or ncores < 1:
        raise RuntimeError(f"failed to get valid num_vectorcore: {ncores!r}")

    stride_jn = jagged.stride(0)
    stride_dk = dense.stride(0)
    stride_on = out.stride(0)

    grid = (ncores,)
    _add_position_embeddings_kernel[grid](
        jagged,
        seq_offsets,
        high_inds,
        dense,
        out,
        B,
        D,
        ncores,
        scale,
        ind_offsets,
        stride_jn,
        stride_dk,
        stride_on,
        SCALE_JAGGED=scale_jagged,
        BLOCK_D=D,
        hfusion_enable_multiple_consumer_fusion=True,
    )
    return out