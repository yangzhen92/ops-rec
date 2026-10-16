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
build_causal_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_causal_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_causal_mask_arbitrary_func_mask_cpu(offsets, total_seqlen):
    """NumPy reference, output [1, 1, total_seqlen, 2].

    语义(单区间, 全局坐标, 半开区间):seq 内每个 query 对 k<=q 可见,即
    af[q] = [seq_start, global_pos + 1]。
    """
    offsets = np.asarray(offsets, dtype=np.int32)
    af = np.zeros((1, 1, total_seqlen, 2), dtype=np.int32)
    positions = np.arange(total_seqlen, dtype=np.int32)
    seq_id = np.searchsorted(offsets[1:] - 1, positions, side="left")
    af[0, 0, :, 0] = offsets[seq_id]  # seq_start
    af[0, 0, :, 1] = positions + 1  # global pos + 1(含自身)
    return af


def test_build_causal_mask_arbitrary_func_two_seq():
    # 两个序列 [3, 2], total_seqlen = 5:seq0 的 k<=q ∈ [0,3), seq1 的 k<=q ∈ [3,5)
    offsets = np.array([0, 3, 5], dtype=np.int32)
    total_seqlen = 5

    y = ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(torch.from_numpy(offsets).npu(), total_seqlen)

    y_np = y.cpu().numpy()
    cpu = build_causal_mask_arbitrary_func_mask_cpu(offsets, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_causal_mask_arbitrary_func result mismatch with CPU reference"

    # 抽查:pos2([0,3])、pos4([3,5])
    assert y_np[0, 0, 2].tolist() == [0, 3], y_np[0, 0, 2]
    assert y_np[0, 0, 4].tolist() == [3, 5], y_np[0, 0, 4]
    print("[torch] build_causal_mask_arbitrary_func (two sequences) example PASSED")


def test_build_causal_mask_arbitrary_func_multi_seq():
    # 三个等长序列 [2, 2, 2], total_seqlen = 6
    offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    total_seqlen = 6

    y = ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(torch.from_numpy(offsets).npu(), total_seqlen)

    y_np = y.cpu().numpy()
    cpu = build_causal_mask_arbitrary_func_mask_cpu(offsets, total_seqlen)
    assert y_np.shape == (1, 1, total_seqlen, 2), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_causal_mask_arbitrary_func multi-seq result mismatch"

    assert y_np[0, 0, 1].tolist() == [0, 2], y_np[0, 0, 1]
    assert y_np[0, 0, 5].tolist() == [4, 6], y_np[0, 0, 5]
    print("[torch] build_causal_mask_arbitrary_func (multi sequences) example PASSED")


if __name__ == "__main__":
    test_build_causal_mask_arbitrary_func_two_seq()
    test_build_causal_mask_arbitrary_func_multi_seq()
