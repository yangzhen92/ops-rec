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
hstu_forward_v1 算子 PyTorch 接口使用样例 — paged 格式

覆盖接口（参数名与 torch schema 一一对应）：
    - hstu_paged            : paged 格式，kv_cache 分页存储（解码场景）

用法:
    python3 test_torch_hstu_forward_paged.py
"""

import torch
import ops_rec

# 与 pytest 保持一致：关闭内部格式，保证输入按 ND 布局送入算子
torch.npu.config.allow_internal_format = False


def test_hstu_paged():
    """paged：kv_cache 分页构造（数据构造参考 pytest test_hstu_paged_graph，已实机验证）。

    注：paged 的 numpy golden 过于复杂，此处仅验证输出 shape 与运行成功；
    数值等价性验证见 pytest（同一输入下 compiled 与 eager 输出一致）。
    """
    torch.npu.set_device(0)
    N, D = 4, 32
    page_size = 32  # tiling 约束：page_size 必须 ∈ {32, 64, 128, 256}
    batch = 2
    # 每个 batch：历史 token 数（已存入 kv_cache）与目标（新生成）token 数
    new_hist_lens = [32, 48]  # 覆盖恰好满页(32)与末页不足(48 % 32 = 16)两种情形
    num_targets = [4, 6]

    seq_lens = [h + t for h, t in zip(new_hist_lens, num_targets)]
    total = sum(seq_lens)
    max_seq_len = max(seq_lens)

    q = torch.randn(total, N, D, dtype=torch.float16).npu()
    k = torch.randn(total, N, D, dtype=torch.float16).npu()
    v = torch.randn(total, N, D, dtype=torch.float16).npu()

    seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
    seq_offset_k = seq_offset.clone()
    seq_offset_t = torch.tensor(
        [0] + [sum(num_targets[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32
    )
    num_target = torch.tensor(num_targets, device="npu", dtype=torch.int32)

    # kv_cache 分页：每个 batch 的历史 KV 按 page_size 分页存放（末页不足时以 last_page_len 标记）
    # 注意：page_size 必须 ∈ {32, 64, 128, 256}（tiling 层 BLOCK_HEIGHT=256 / MIN_PAGE_SIZE=32 约束）
    num_pages_per_batch = [(h + page_size - 1) // page_size for h in new_hist_lens]
    page_ids = list(range(sum(num_pages_per_batch)))
    page_offsets = [0] + [sum(num_pages_per_batch[: i + 1]) for i in range(batch)]
    last_page_len = [h % page_size if h % page_size > 0 else page_size for h in new_hist_lens]

    kv_cache = torch.randn(sum(num_pages_per_batch), 2, page_size, N, D, device="npu", dtype=torch.float16)
    page_offsets = torch.tensor(page_offsets, device="npu", dtype=torch.int32)
    page_ids = torch.tensor(page_ids, device="npu", dtype=torch.int32)
    last_page_len = torch.tensor(last_page_len, device="npu", dtype=torch.int32)

    # 注意：num_target 定义时 target_group_size 必须 > 0（torch 扩展层约束）
    y = ops_rec.ascendc.attention.hstu_paged(
        q,
        k,
        v,
        kv_cache,
        mask=None,
        attn_bias=None,
        mask_type=0,
        max_seq_len=max_seq_len,
        max_seq_len_k=max_seq_len,
        silu_scale=1.0 / max_seq_len,
        seq_offset=seq_offset,
        seq_offset_k=seq_offset_k,
        seq_offset_t=seq_offset_t,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_len=last_page_len,
        num_target=num_target,
        target_group_size=1,
    )

    assert y.shape == (total, N, D), f"unexpected output shape {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "output contains non-finite values"
    print(f"[torch] hstu_paged output shape: {tuple(y.shape)}")
    print("[torch] hstu_paged torch example PASSED（shape/运行成功校验，数值等价性见 pytest）")


if __name__ == "__main__":
    test_hstu_paged()
    print("[torch] hstu_paged torch example done")
