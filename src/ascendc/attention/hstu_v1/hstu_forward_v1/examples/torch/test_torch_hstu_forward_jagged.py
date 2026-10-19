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
hstu_forward_v1 算子 PyTorch 接口使用样例 — jagged 格式

hstu_jagged 为 torch OpOverloadPacket（equal / delta 两个 overload 注册为同名接口），
调用 ops_rec.ascendc.attention.hstu_jagged(...) 时按参数数量/类型自动分派：
    - equal 语义（q 与 k/v 共用同一 seq_offset，段长一致）：不传 max_seq_len_k / seq_offset_k
    - delta 语义（q 与 k/v 使用不同 seq_offset / seq_offset_k，段长可不同）：多传 max_seq_len_k / seq_offset_k

示例主体直接调用 hstu_jagged(...)（自动识别），末尾保留 equal / delta 的简短基本示例。

用法:
    python3 test_torch_hstu_forward_jagged.py
"""

import torch
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def test_hstu_jagged_equal():
    """简短基本示例：hstu_jagged.equal（q 与 k/v 共用同一 seq_offset，段长一致）。"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    silu_scale = 1.0 / S

    q = torch.randn(total, N, D, dtype=torch.float16).npu()
    k = torch.randn(total, N, D, dtype=torch.float16).npu()
    v = torch.randn(total, N, D, dtype=torch.float16).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.equal(q, k, v, None, None, 0, S, silu_scale, seq_offset)

    torch.npu.synchronize()
    print(f"[torch] hstu_jagged.equal output shape: {tuple(y.shape)}")


def test_hstu_jagged_delta():
    """简短基本示例：hstu_jagged.delta（q 与 k/v 使用不同 seq_offset / seq_offset_k，段长可不同）。"""
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens_q = [4, 6]
    seq_lens_k = [6, 8]
    batch = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    s_q = max(seq_lens_q)
    s_k = max(seq_lens_k)
    silu_scale = 1.0 / s_q

    q = torch.randn(total_q, N, D, dtype=torch.float16).npu()
    k = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    v = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens_q[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = torch.tensor([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    y = ops_rec.ascendc.attention.hstu_jagged.delta(
        q, k, v, None, None, 0, s_q, s_k, silu_scale, seq_offset, seq_offset_k
    )

    torch.npu.synchronize()
    print(f"[torch] hstu_jagged.delta output shape: {tuple(y.shape)}")


def test_hstu_jagged_dispatch():
    """主体示例：直接调用 hstu_jagged(...)，torch 按参数自动分派到 equal / delta overload。

    equal 语义（参数到 seq_offset 为止）：
        hstu_jagged(q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale, seq_offset)
    delta 语义（额外传 max_seq_len_k / seq_offset_k）：
        hstu_jagged(q, k, v, mask, attn_bias, mask_type, max_seq_len, max_seq_len_k,
                    silu_scale, seq_offset, seq_offset_k)
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    seq_lens = [4, 6]
    batch = len(seq_lens)
    total = sum(seq_lens)
    S = max(seq_lens)
    silu_scale = 1.0 / S

    q = torch.randn(total, N, D, dtype=torch.float16).npu()
    k = torch.randn(total, N, D, dtype=torch.float16).npu()
    v = torch.randn(total, N, D, dtype=torch.float16).npu()
    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)

    # equal 语义（不传 max_seq_len_k / seq_offset_k）
    y = ops_rec.ascendc.attention.hstu_jagged(q, k, v, None, None, 0, S, silu_scale, seq_offset)
    torch.npu.synchronize()
    print(f"[torch] hstu_jagged (equal) output shape: {tuple(y.shape)}")

    # delta 语义（多传 max_seq_len_k / seq_offset_k）
    seq_lens_k = [6, 8]
    total_k = sum(seq_lens_k)
    k_d = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    v_d = torch.randn(total_k, N, D, dtype=torch.float16).npu()
    seq_offset_k = torch.tensor([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    y_d = ops_rec.ascendc.attention.hstu_jagged(q, k_d, v_d, None, None, 0, S, 8, silu_scale, seq_offset, seq_offset_k)
    torch.npu.synchronize()
    print(f"[torch] hstu_jagged (delta) output shape: {tuple(y_d.shape)}")


if __name__ == "__main__":
    test_hstu_jagged_dispatch()
    test_hstu_jagged_equal()
    test_hstu_jagged_delta()
    print("[torch] hstu_jagged torch examples done")
