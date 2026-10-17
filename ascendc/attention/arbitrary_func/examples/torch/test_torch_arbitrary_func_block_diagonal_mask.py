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
build_block_diagonal_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_block_diagonal_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_block_diagonal_mask_arbitrary_func_cpu(offsets, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 2].

    语义: 每个 query 仅能看到自身所在序列的所有 key
    af[q] = [seq_start, seq_end]
    """
    batch_size = len(offsets) - 1
    af = np.zeros((1, 1, total_seqlen, 2), dtype=np.int32)
    for pos in range(total_seqlen):
        lo, hi = 0, batch_size
        while lo < hi:
            mid = (lo + hi) >> 1
            if offsets[mid + 1] <= pos:
                lo = mid + 1
            else:
                hi = mid
        seq_id = lo
        seq_start = int(offsets[seq_id])
        seq_end = int(offsets[seq_id + 1])
        af[0, 0, pos, 0] = seq_start
        af[0, 0, pos, 1] = seq_end
    return af


def test_build_block_diagonal_mask_arbitrary_func():
    offsets = np.array([0, 3, 7], dtype=np.int32)
    total_seqlen = 7

    y = ops_rec.ascendc.attention.build_block_diagonal_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(),
        total_seqlen,
    )

    y_np = y.cpu().numpy()
    cpu = build_block_diagonal_mask_arbitrary_func_cpu(offsets, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_block_diagonal_mask_arbitrary_func result mismatch with CPU reference"

    # 抽查: pos0 (Seq0) expect [0, 3], pos3 (Seq1) expect [3, 7]
    assert y_np[0, 0, 0].tolist() == [0, 3], y_np[0, 0, 0]
    assert y_np[0, 0, 3].tolist() == [3, 7], y_np[0, 0, 3]
    assert y_np[0, 0, 6].tolist() == [3, 7], y_np[0, 0, 6]
    print("[torch] build_block_diagonal_mask_arbitrary_func example PASSED")


if __name__ == "__main__":
    test_build_block_diagonal_mask_arbitrary_func()
