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
create_q2k_block_sparse_from_func 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_create_q2k_block_sparse_from_func.py

Q2K（正向）块稀疏张量生成：根据函数编码张量 ``func_tensor``，为每个 (b, h, q_block)
并行计算其覆盖的 kv_block 列表，输出 mask/full 块数与块索引表，供
``LinearBlockSparseTensors`` / FlexAttention block-mask 使用。

``func_tensor`` 布局为 [B, H, func_q_len, n_func]（n_func 为偶数），每个 q token
的合法 kv 区间为 ``[F_{2i}, F_{2i+1})`` 的并集。每个 (q_block, kv_block) 块被分类为：

* FULL  —— 块内所有 (q_token, kv 位置) 均合法
* MASK  —— 块内部分位置合法
* EMPTY —— 块内无合法位置

``block_idx`` 每行内 FULL 块 id 从左往右写、MASK 块 id 从右往左写。``check_q_boundary``
为 true 时，部分 q_block（不足 q_block_size）不能判为 FULL。
"""

import torch
import numpy as np
import ops_rec

# 状态常量（与 kernel 中 STATE_EMPTY/STATE_MASK/STATE_FULL 一致）
STATE_EMPTY = 0
STATE_MASK = 1
STATE_FULL = 2


def get_token_kv_block_state(func_row, kv_start, kv_end):
    """判断单个 q_token 与 [kv_start, kv_end) 块的状态。

    func_row: [n_func] 的区间编码，合法区间为 [F_{2i}, F_{2i+1})。
    返回 STATE_FULL / STATE_MASK / STATE_EMPTY。
    """
    fully_covered = False
    has_overlap = False
    for i in range(0, len(func_row), 2):
        s, e = int(func_row[i]), int(func_row[i + 1])
        if e <= s:
            continue
        if s <= kv_start and e >= kv_end:
            fully_covered = True
        elif s < kv_end and e > kv_start:
            has_overlap = True
    if fully_covered:
        return STATE_FULL
    if has_overlap:
        return STATE_MASK
    return STATE_EMPTY


def merge_token_states(states):
    """将 q_block 内各 token 的状态按位或合并为块状态（与 kernel 位掩码归约一致）。"""
    state_union = 0
    for st in states:
        state_union |= 1 << st
    if state_union == 0b001:
        return STATE_EMPTY
    if state_union == 0b100:
        return STATE_FULL
    return STATE_MASK


def create_q2k_block_sparse_from_func_cpu(func_tensor, q_len, kv_len, q_block_size, kv_block_size, check_q_boundary):
    func_tensor = np.asarray(func_tensor, dtype=np.int32)
    B, H, func_q_len, n_func = func_tensor.shape
    num_q_blocks = (q_len + q_block_size - 1) // q_block_size
    num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size

    mask_block_cnt = np.zeros((B, H, num_q_blocks), dtype=np.int32)
    full_block_cnt = np.zeros((B, H, num_q_blocks), dtype=np.int32)
    block_idx = np.full((B, H, num_q_blocks, num_kv_blocks), -1, dtype=np.int32)

    for b in range(B):
        for h in range(H):
            for qb in range(num_q_blocks):
                q_start = qb * q_block_size
                q_end = min(q_start + q_block_size, q_len)
                is_q_block_full = (q_end - q_start) == q_block_size
                full_count = 0
                mask_count = 0
                for kvb in range(num_kv_blocks):
                    kv_start = kvb * kv_block_size
                    kv_end = min(kv_start + kv_block_size, kv_len)
                    is_kv_block_full = (kv_end - kv_start) == kv_block_size
                    states = [
                        get_token_kv_block_state(func_tensor[b, h, q], kv_start, kv_end) for q in range(q_start, q_end)
                    ]
                    kv_block_state = merge_token_states(states)
                    can_be_full = (not check_q_boundary or is_q_block_full) and is_kv_block_full
                    if kv_block_state == STATE_FULL and can_be_full:
                        block_idx[b, h, qb, full_count] = kvb
                        full_count += 1
                    elif kv_block_state in (STATE_FULL, STATE_MASK):
                        block_idx[b, h, qb, num_kv_blocks - 1 - mask_count] = kvb
                        mask_count += 1
                full_block_cnt[b, h, qb] = full_count
                mask_block_cnt[b, h, qb] = mask_count

    return mask_block_cnt, full_block_cnt, block_idx


def check_block_idx(npu_idx, cpu_idx, full_cnt, mask_cnt):
    """校验 block_idx：FULL 块在 [0, full_cnt)，MASK 块在 [max-mask_cnt, max)。"""
    B, H, num_blocks, max_blocks = npu_idx.shape
    for b in range(B):
        for h in range(H):
            for blk in range(num_blocks):
                fcnt = int(full_cnt[b, h, blk])
                mcnt = int(mask_cnt[b, h, blk])
                if fcnt > 0:
                    assert torch.equal(
                        npu_idx[b, h, blk, :fcnt].cpu(),
                        torch.from_numpy(cpu_idx[b, h, blk, :fcnt]),
                    ), f"full block_idx mismatch at (b={b}, h={h}, q_block={blk})"
                if mcnt > 0:
                    assert torch.equal(
                        npu_idx[b, h, blk, max_blocks - mcnt :].cpu(),
                        torch.from_numpy(cpu_idx[b, h, blk, max_blocks - mcnt :]),
                    ), f"mask block_idx mismatch at (b={b}, h={h}, q_block={blk})"


def test_create_q2k_block_sparse_from_func():
    """验证 create_q2k_block_sparse_from_func 算子的前向推理结果。"""
    B, H = 1, 1
    Q_LEN, KV_LEN = 8, 8
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 4, 4
    NUM_Q_BLOCKS = (Q_LEN + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    NUM_KV_BLOCKS = (KV_LEN + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    # func_tensor: [B, H, func_q_len, n_func] = [1, 1, 8, 2]（n_func=2）
    # 每个 token 的合法 kv 区间 [F0, F1)：
    #   token0/1: [0, 8)   token2/3: [2, 8)
    #   token4/5: [4, 8)   token6/7: [0, 4)
    func_tensor = np.array(
        [
            [0, 8],
            [0, 8],
            [2, 8],
            [2, 8],
            [4, 8],
            [4, 8],
            [0, 4],
            [0, 4],
        ],
        dtype=np.int32,
    ).reshape(B, H, Q_LEN, 2)

    ft = torch.from_numpy(func_tensor).npu()
    out = ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
        ft, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, True
    )
    assert len(out) == 3, f"expected 3-tuple, got {len(out)}"
    npu_mask_cnt, npu_full_cnt, npu_block_idx = out

    # 手动计算验证
    cpu_mask_cnt, cpu_full_cnt, cpu_block_idx = create_q2k_block_sparse_from_func_cpu(
        func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True
    )

    assert npu_mask_cnt.dtype == torch.int32 and npu_full_cnt.dtype == torch.int32
    assert npu_block_idx.dtype == torch.int32
    assert tuple(npu_mask_cnt.shape) == (B, H, NUM_Q_BLOCKS)
    assert tuple(npu_full_cnt.shape) == (B, H, NUM_Q_BLOCKS)
    assert tuple(npu_block_idx.shape) == (B, H, NUM_Q_BLOCKS, NUM_KV_BLOCKS)

    assert torch.equal(npu_mask_cnt.cpu(), torch.from_numpy(cpu_mask_cnt)), "mask_block_cnt mismatch"
    assert torch.equal(npu_full_cnt.cpu(), torch.from_numpy(cpu_full_cnt)), "full_block_cnt mismatch"
    check_block_idx(npu_block_idx, cpu_block_idx, cpu_full_cnt, cpu_mask_cnt)

    print(
        f"[torch] shapes: mask_cnt={tuple(npu_mask_cnt.shape)} "
        f"full_cnt={tuple(npu_full_cnt.shape)} block_idx={tuple(npu_block_idx.shape)}"
    )
    print(f"[torch] mask_block_cnt={npu_mask_cnt.cpu().tolist()}")
    print(f"[torch] full_block_cnt={npu_full_cnt.cpu().tolist()}")
    print(f"[torch] block_idx={npu_block_idx.cpu().tolist()}")
    print("[torch] create_q2k_block_sparse_from_func torch example PASSED")


if __name__ == "__main__":
    test_create_q2k_block_sparse_from_func()
