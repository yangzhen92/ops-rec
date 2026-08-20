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
hstu_forward_v1 算子 PyTorch 接口使用样例 — 量化（fp8）场景

覆盖场景（参数名与 torch schema 一一对应）：
    - hstu_jagged.equal（jagged 格式，Atlas 950 专用）：q/k/v 为 fp8_e4m3fn，输出为 float16

用法:
    python3 test_torch_hstu_forward_fp8.py
"""

import torch
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def test_hstu_jagged_fp8():
    """运行 hstu_jagged.equal fp8 量化前向样例。

    仅 Atlas 950（A5）支持 fp8_e4m3fn 输入，输出为 float16。
    量化权重一般来自模型量化（如 fp8 权重 + 激活），此处用随机数据演示接口调用。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    silu_scale = 1.0 / S

    q = torch.randn(total, N, D).to(dtype=torch.float8_e4m3fn).npu()
    k = torch.randn(total, N, D).to(dtype=torch.float8_e4m3fn).npu()
    v = torch.randn(total, N, D).to(dtype=torch.float8_e4m3fn).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.equal(q, k, v, None, None, 0, S, silu_scale, seq_offset)

    torch.npu.synchronize()
    assert y.dtype == torch.float16, f"fp8 输入应输出 float16，实际 {y.dtype}"
    assert y.shape == (total, N, D), f"unexpected output shape {tuple(y.shape)}"
    print(f"[torch] hstu_jagged.equal(fp8) input dtype: {q.dtype}, output dtype: {y.dtype}, shape: {tuple(y.shape)}")
    print("[torch] hstu_jagged fp8 torch example PASSED")


if __name__ == "__main__":
    test_hstu_jagged_fp8()
    print("[torch] hstu_jagged fp8 torch example done")
