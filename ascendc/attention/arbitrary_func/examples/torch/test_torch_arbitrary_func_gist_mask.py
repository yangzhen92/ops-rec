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
build_gist_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_gist_mask.py
"""

import numpy as np
import torch
import ops_rec


def add_interval(intervals, begin, end, seq_start, seq_end, max_intervals=2):
    begin = max(begin, seq_start)
    end = min(end, seq_end)
    if begin >= end:
        return
    if intervals and intervals[-1][1] == begin:
        intervals[-1] = (intervals[-1][0], end)
    else:
        if len(intervals) >= max_intervals:
            return
        intervals.append((begin, end))


def build_gist_mask_arbitrary_func_cpu(offsets, pre_doc_lens, gist_starts, gist_ends, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 4]."""
    batch_size = len(offsets) - 1
    af = np.zeros((1, 1, total_seqlen, 4), dtype=np.int32)
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
        pos_local = pos - seq_start
        p = int(min(max(pre_doc_lens[seq_id], 0), seq_end - seq_start))
        g0 = int(min(max(gist_starts[seq_id], p), seq_end - seq_start))
        g1 = int(min(max(gist_ends[seq_id], g0), seq_end - seq_start))

        intervals = []
        if pos_local < p:
            add_interval(intervals, seq_start, pos + 1, seq_start, seq_end)
        elif pos_local < g0:
            add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
            add_interval(intervals, seq_start + p, pos + 1, seq_start, seq_end)
        elif pos_local < g1:
            add_interval(intervals, seq_start, pos + 1, seq_start, seq_end)
        else:
            add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
            add_interval(intervals, seq_start + g0, seq_start + g1, seq_start, seq_end)
            add_interval(intervals, seq_start + g1, pos + 1, seq_start, seq_end)

        while len(intervals) < 2:
            last = intervals[-1][1] if intervals else seq_start
            intervals.append((last, last))
        af[0, 0, pos] = np.asarray(intervals[:2], dtype=np.int32).reshape(-1)
    return af


def test_build_gist_mask_arbitrary_func():
    offsets = np.array([0, 10], dtype=np.int32)
    pre_doc_lens = np.array([3], dtype=np.int32)
    gist_starts = np.array([5], dtype=np.int32)
    gist_ends = np.array([7], dtype=np.int32)
    total_seqlen = 10

    y = ops_rec.ascendc.attention.build_gist_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(),
        torch.from_numpy(pre_doc_lens).npu(),
        torch.from_numpy(gist_starts).npu(),
        torch.from_numpy(gist_ends).npu(),
        total_seqlen,
    )

    y_np = y.cpu().numpy()
    cpu = build_gist_mask_arbitrary_func_cpu(offsets, pre_doc_lens, gist_starts, gist_ends, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 4), f"shape {y_np.shape}"

    # 抽查: pos0(pre_doc) expect [0,1,1,1] (1区间+end填充)
    assert y_np[0, 0, 0].tolist() == [0, 1, 1, 1], y_np[0, 0, 0]
    # pos4(bridge): [0,3)∪[3,5) 合并为 [0,5), fill=[5,5]
    assert y_np[0, 0, 4].tolist() == [0, 5, 5, 5], y_np[0, 0, 4]
    # pos6(gist) expect [0,7,7,7] (1区间+end填充)
    assert y_np[0, 0, 6].tolist() == [0, 7, 7, 7], y_np[0, 0, 6]
    # pos7(post-gist): [5,7)∪[7,8) 合并为 [5,8)
    assert y_np[0, 0, 7].tolist() == [0, 3, 5, 8], y_np[0, 0, 7]
    # pos8(post-gist): [5,7)∪[7,9) 合并为 [5,9)
    assert y_np[0, 0, 8].tolist() == [0, 3, 5, 9], y_np[0, 0, 8]
    # pos9(post-gist): [5,7)∪[7,10) 合并为 [5,10)
    assert y_np[0, 0, 9].tolist() == [0, 3, 5, 10], y_np[0, 0, 9]
    print("[torch] build_gist_mask_arbitrary_func example PASSED")


if __name__ == "__main__":
    test_build_gist_mask_arbitrary_func()
