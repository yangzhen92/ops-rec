#!/usr/bin/env python3
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

"""
build_hstu_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_hstu_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_hstu_mask_arbitrary_func_mask_cpu(offsets_i32, ctx_lens, tgt_lens, target_group_size, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 4]."""
    arbitrary_func = np.zeros((1, 1, total_seqlen, 4), dtype=np.int32)
    af = arbitrary_func[0, 0, :, :]

    positions = np.arange(total_seqlen, dtype=np.int32)
    seq_id = np.searchsorted(offsets_i32[1:] - 1, positions, side='left')

    seq_start = offsets_i32[seq_id]
    seq_end = offsets_i32[seq_id + 1]
    seq_len = seq_end - seq_start

    ctx_len = ctx_lens[seq_id]
    tgt_len = tgt_lens[seq_id]
    hist_len = seq_len - ctx_len - tgt_len

    pos_in_seq = positions - seq_start
    ctx_end = ctx_len
    hist_end = ctx_len + hist_len
    base = seq_start + hist_end

    is_ctx = pos_in_seq < ctx_end
    is_hist = (pos_in_seq >= ctx_end) & (pos_in_seq < hist_end)
    is_tgt = pos_in_seq >= hist_end

    af[is_ctx, 0] = seq_start[is_ctx]
    af[is_ctx, 1] = base[is_ctx]

    af[is_hist, 0] = seq_start[is_hist]
    af[is_hist, 1] = (positions + 1)[is_hist]

    tgt_idx = pos_in_seq - hist_end
    group_start = (tgt_idx // target_group_size) * target_group_size
    af[is_tgt, 0] = seq_start[is_tgt]
    af[is_tgt, 1] = base[is_tgt]
    af[is_tgt, 2] = (base + group_start)[is_tgt]
    af[is_tgt, 3] = (base + tgt_idx + 1)[is_tgt]

    return arbitrary_func


def test_build_hstu_mask_arbitrary_func():
    # 两个序列：[2 ctx, 1 tgt], [3 ctx, 2 tgt]，无 history，total_seqlen = 8
    offsets = np.array([0, 3, 8], dtype=np.int32)
    ctx_lens = np.array([2, 3], dtype=np.int32)
    tgt_lens = np.array([1, 2], dtype=np.int32)
    group_size = 2
    total_seqlen = 8

    y = ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(),
        torch.from_numpy(ctx_lens).npu(),
        torch.from_numpy(tgt_lens).npu(),
        group_size,
        total_seqlen,
    )

    y_np = y.cpu().numpy()
    cpu = build_hstu_mask_arbitrary_func_mask_cpu(offsets, ctx_lens, tgt_lens, group_size, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 4), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_hstu_mask_arbitrary_func result mismatch with CPU reference"
    print("[torch] build_hstu_mask_arbitrary_func torch example PASSED")


if __name__ == "__main__":
    test_build_hstu_mask_arbitrary_func()
