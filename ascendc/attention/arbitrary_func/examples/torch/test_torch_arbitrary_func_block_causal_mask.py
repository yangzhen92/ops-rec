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
build_block_causal_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_block_causal_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_block_causal_mask_arbitrary_func_cpu(offsets_i32, block_sizes, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 2]."""
    arbitrary_func = np.zeros((1, 1, total_seqlen, 2), dtype=np.int32)
    af = arbitrary_func[0, 0, :, :]

    positions = np.arange(total_seqlen, dtype=np.int32)

    seq_id = np.searchsorted(offsets_i32[1:] - 1, positions, side='left')

    seq_start = offsets_i32[seq_id]
    seq_end = offsets_i32[seq_id + 1]
    seq_len = seq_end - seq_start

    block_size = block_sizes[seq_id]
    block_size = np.where(block_size > 0, block_size, 1)

    pos_in_seq = positions - seq_start
    block_start = (pos_in_seq // block_size) * block_size
    block_end = np.minimum(block_start + block_size, seq_len)

    af[:, 0] = seq_start
    af[:, 1] = seq_start + block_end

    return arbitrary_func


def test_build_block_causal_mask_arbitrary_func():
    offsets = np.array([0, 5, 10], dtype=np.int32)
    block_sizes = np.array([2, 3], dtype=np.int32)
    total_seqlen = 10

    y = ops_rec.ascendc.attention.build_block_causal_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(),
        torch.from_numpy(block_sizes).npu(),
        total_seqlen,
    )

    y_np = y.cpu().numpy()
    cpu = build_block_causal_mask_arbitrary_func_cpu(offsets, block_sizes, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), (
        f"build_block_causal_mask_arbitrary_func result mismatch.\n"
        f"NPU output:\n{y_np[0, 0]}\nCPU reference:\n{cpu[0, 0]}"
    )
    print("[torch] build_block_causal_mask_arbitrary_func torch example PASSED")


if __name__ == "__main__":
    test_build_block_causal_mask_arbitrary_func()
