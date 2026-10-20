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
# pylint: disable=duplicate-code
"""
build_sliding_window_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_sliding_window_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_sliding_window_mask_arbitrary_func_mask_cpu(offsets, window_sizes, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 2].

    语义(单区间, 全局坐标, 半开区间):seq 内每个 query q 只见窗口大小为 window 的最近几个 key,
    即 af[q] = [max(seq_start, q - window + 1), q + 1),window 取 max(win, 1)。
    """
    offsets = np.asarray(offsets, dtype=np.int32)
    window_sizes = np.asarray(window_sizes, dtype=np.int32)
    af = np.zeros((1, 1, total_seqlen, 2), dtype=np.int32)
    positions = np.arange(total_seqlen, dtype=np.int32)
    seq_id = np.searchsorted(offsets[1:] - 1, positions, side="left")
    seq_start = offsets[seq_id]
    pos = positions - seq_start
    window = np.maximum(window_sizes[seq_id], 1)
    af[0, 0, :, 0] = seq_start + np.maximum(pos - window + 1, 0)
    af[0, 0, :, 1] = positions + 1
    return af


def test_build_sliding_window_mask_arbitrary_func_two_seq():
    # 两个序列 [3, 2], total_seqlen = 5;seq0 窗口=2, seq1 窗口=1
    #   seq0: q0->[0,1) q1->[0,2) q2->[1,3);seq1: q3->[3,4) q4->[4,5)
    offsets = np.array([0, 3, 5], dtype=np.int32)
    window_sizes = np.array([2, 1], dtype=np.int32)
    total_seqlen = 5

    y = ops_rec.ascendc.attention.build_sliding_window_mask_arbitrary_func(torch.from_numpy(offsets).npu(),
                                                                           torch.from_numpy(window_sizes).npu(),
                                                                           total_seqlen)

    y_np = y.cpu().numpy()
    cpu = build_sliding_window_mask_arbitrary_func_mask_cpu(offsets, window_sizes, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_sliding_window_mask_arbitrary_func result mismatch with CPU reference"

    # 抽查:pos1([0,2])、pos2([1,3])、pos4([4,5])
    assert y_np[0, 0, 1].tolist() == [0, 2], y_np[0, 0, 1]
    assert y_np[0, 0, 2].tolist() == [1, 3], y_np[0, 0, 2]
    assert y_np[0, 0, 4].tolist() == [4, 5], y_np[0, 0, 4]
    print("[torch] build_sliding_window_mask_arbitrary_func (two sequences) example PASSED")


def test_build_sliding_window_mask_arbitrary_func_multi_seq():
    # 三个序列 [2, 2, 2], total_seqlen = 6;窗口 [1, 3, 2]
    #   seq0: q0->[0,1) q1->[1,2);seq1: q2->[2,3) q3->[2,4)
    #   seq2: q4->[4,5) q5->[4,6)
    offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    window_sizes = np.array([1, 3, 2], dtype=np.int32)
    total_seqlen = 6

    y = ops_rec.ascendc.attention.build_sliding_window_mask_arbitrary_func(torch.from_numpy(offsets).npu(),
                                                                           torch.from_numpy(window_sizes).npu(),
                                                                           total_seqlen)

    y_np = y.cpu().numpy()
    cpu = build_sliding_window_mask_arbitrary_func_mask_cpu(offsets, window_sizes, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_sliding_window_mask_arbitrary_func multi-seq result mismatch"

    # seq1 (起点 2, 窗口 3):q3(全局 pos3) 应为 [max(2, 3-3+1), 4) = [2, 4)
    assert y_np[0, 0, 3].tolist() == [2, 4], y_np[0, 0, 3]
    # seq2 末行(全局 pos5, 窗口 2):[max(4, 5-2+1), 6) = [4, 6)
    assert y_np[0, 0, 5].tolist() == [4, 6], y_np[0, 0, 5]
    print("[torch] build_sliding_window_mask_arbitrary_func (multi sequences) example PASSED")


if __name__ == "__main__":
    test_build_sliding_window_mask_arbitrary_func_two_seq()
    test_build_sliding_window_mask_arbitrary_func_multi_seq()