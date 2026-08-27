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
build_sla_mask_arbitrary_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_arbitrary_func_sla_mask.py
"""

import numpy as np
import torch
import ops_rec


def build_sla_mask_arbitrary_func_mask_cpu(offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets):
    """NumPy reference, output [1, 1, total_seqlen, 4].

    语义（v2 af，全局坐标，半开区间）：
      history: c0=min(pos_local+1, eff_k2) / c1=max(eff_k2, pos_local-sla_k1+1) / c2=pos_local+1
      target : c0=c1=c2=H（只见 [0,H)，连自身不可见）
      af[q] = [S, S+c0, S+c1, S+c2]
    """
    batch_size = len(offsets) - 1
    eff_k2 = max(sla_k2, contextual_seq_len)
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
        seq_len = int(offsets[seq_id + 1]) - seq_start
        pos_local = pos - seq_start
        h = max(seq_len - int(num_targets[seq_id]), 0)
        if pos_local < h:  # history
            c0 = min(pos_local + 1, eff_k2)
            c1 = max(eff_k2, pos_local - sla_k1 + 1)
            c2 = pos_local + 1
        else:              # target：只见 [0,h)，连自身不可见
            c0 = c1 = c2 = h
        af[0, 0, pos, 0] = seq_start
        af[0, 0, pos, 1] = seq_start + c0
        af[0, 0, pos, 2] = seq_start + c1
        af[0, 0, pos, 3] = seq_start + c2
    return af


def test_build_sla_mask_arbitrary_func_with_target():
    # 两个序列：[6 len, 2 tgt], [4 len, 1 tgt]，K1=2 / K2=3 / ctx=0，total_seqlen = 10
    #   seq0 的 history 边界 H=4；seq1 的 H=3
    offsets = np.array([0, 6, 10], dtype=np.int32)
    num_targets = np.array([2, 1], dtype=np.int32)
    sla_k1, sla_k2, contextual_seq_len, total_seqlen = 2, 3, 0, 10

    y = ops_rec.ascendc.attention.build_sla_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(),
        total_seqlen,
        sla_k1,
        sla_k2,
        contextual_seq_len,
        torch.from_numpy(num_targets).npu(),
    )

    y_np = y.cpu().numpy()
    cpu = build_sla_mask_arbitrary_func_mask_cpu(
        offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets)
    assert y_np.shape == (1, 1, total_seqlen, 4), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), "build_sla_mask_arbitrary_func result mismatch with CPU reference"

    # 抽查两行：seq0 的 target（pos4，expect [0,4,4,4]）、seq1 的 target（pos9，expect [6,9,9,9]）
    assert y_np[0, 0, 4].tolist() == [0, 4, 4, 4], y_np[0, 0, 4]
    assert y_np[0, 0, 9].tolist() == [6, 9, 9, 9], y_np[0, 0, 9]
    print("[torch] build_sla_mask_arbitrary_func (with target) example PASSED")


def test_build_sla_mask_arbitrary_func_default_nt():
    # num_targets 缺省 = 全 0，全部为 history token
    offsets = np.array([0, 5, 8], dtype=np.int32)
    sla_k1, sla_k2, contextual_seq_len, total_seqlen = 2, 3, 0, 8
    num_targets = np.zeros(len(offsets) - 1, dtype=np.int32)

    y = ops_rec.ascendc.attention.build_sla_mask_arbitrary_func(
        torch.from_numpy(offsets).npu(), total_seqlen, sla_k1, sla_k2, contextual_seq_len)  # 不传 num_targets

    y_np = y.cpu().numpy()
    cpu = build_sla_mask_arbitrary_func_mask_cpu(
        offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets)
    assert np.array_equal(y_np, cpu), "default num_targets example mismatch with CPU reference"
    print("[torch] build_sla_mask_arbitrary_func (default num_targets) example PASSED")


if __name__ == "__main__":
    test_build_sla_mask_arbitrary_func_with_target()
    test_build_sla_mask_arbitrary_func_default_nt()
