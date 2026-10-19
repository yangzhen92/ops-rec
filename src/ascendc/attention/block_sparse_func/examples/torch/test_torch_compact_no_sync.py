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
compact_no_sync 算子 PyTorch 接口使用样例

用法:
    python3 test_torch_compact_no_sync.py

该算子是 flash-attn 中 ``compact_block_idx`` 的无同步替代实现。输入为每个
batch/head/block 的 mask 块数与 full 块数，以及填充后的 ``block_idx`` 查找表，
输出 CSR 风格的 offset 与压缩后的 index 缓冲，供 ``LinearBlockSparseTensors`` 使用。
"""

import torch
import numpy as np
import ops_rec


def compact_no_sync_cpu(mask_cnt, full_cnt, block_idx):
    """NumPy 参考实现，与 NPU 算子的输出对齐。

    返回 6 元组，顺序与 NPU 算子一致：
        (mask_block_cnt, mask_offset, mask_compact,
         full_block_cnt, full_offset, full_compact)
    """
    mask_cnt = np.asarray(mask_cnt, dtype=np.int32)
    full_cnt = np.asarray(full_cnt, dtype=np.int32)
    block_idx = np.asarray(block_idx, dtype=np.int32)
    B, H, num_blocks = mask_cnt.shape
    n_flat = B * H * num_blocks
    max_blocks = block_idx.shape[3]

    mask_offset = np.zeros(n_flat + 1, dtype=np.int32)
    full_offset = np.zeros(n_flat + 1, dtype=np.int32)
    flat_mask = mask_cnt.reshape(-1)
    flat_full = full_cnt.reshape(-1)
    flat_bidx = block_idx.reshape(n_flat, max_blocks)

    for i in range(n_flat):
        mask_offset[i + 1] = mask_offset[i] + int(flat_mask[i])
        full_offset[i + 1] = full_offset[i] + int(flat_full[i])

    mask_idx = np.zeros(int(mask_offset[-1]), dtype=np.int32)
    full_idx = np.zeros(int(full_offset[-1]), dtype=np.int32)
    for i in range(n_flat):
        mcnt, fcnt = int(flat_mask[i]), int(flat_full[i])
        row = flat_bidx[i]
        # full_compact 取每行前 fcnt 个
        for j in range(fcnt):
            full_idx[full_offset[i] + j] = row[j]
        # mask_compact 取每行后 mcnt 个（逆序）
        for j in range(mcnt):
            mask_idx[mask_offset[i] + j] = row[max_blocks - 1 - j]

    return (
        mask_cnt.astype(np.int32, copy=True),
        mask_offset,
        mask_idx,
        full_cnt.astype(np.int32, copy=True),
        full_offset,
        full_idx,
    )


def test_compact_no_sync():
    """验证 compact_no_sync 算子的前向推理结果。"""
    B, H, num_blocks, max_blocks = 2, 1, 3, 4
    rng = np.random.default_rng(seed=0)

    mask_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    full_cnt = rng.integers(0, max_blocks + 1, size=(B, H, num_blocks), dtype=np.int32)
    block_idx = rng.integers(0, 1000, size=(B, H, num_blocks, max_blocks), dtype=np.int32)

    mc = torch.from_numpy(mask_cnt).npu()
    fc = torch.from_numpy(full_cnt).npu()
    bi = torch.from_numpy(block_idx).npu()

    out = ops_rec.ascendc.attention.compact_no_sync(mc, fc, bi)
    assert len(out) == 6, f"expected 6-tuple, got {len(out)}"
    npu_mc, npu_mo, npu_mcp, npu_fc, npu_fo, npu_fcp = out

    # 手动计算验证
    cpu_mc, cpu_mo, cpu_mcp, cpu_fc, cpu_fo, cpu_fcp = compact_no_sync_cpu(mask_cnt, full_cnt, block_idx)

    assert npu_mc.dtype == torch.int32 and npu_fc.dtype == torch.int32
    assert torch.equal(npu_mc.cpu(), torch.from_numpy(cpu_mc)), "mask_block_cnt mismatch"
    assert torch.equal(npu_fc.cpu(), torch.from_numpy(cpu_fc)), "full_block_cnt mismatch"
    assert torch.equal(npu_mo.cpu(), torch.from_numpy(cpu_mo)), "mask_offset mismatch"
    assert torch.equal(npu_fo.cpu(), torch.from_numpy(cpu_fo)), "full_offset mismatch"

    # 压缩缓冲尾端被过度分配，仅校验有效前缀
    n_mask = int(cpu_mo[-1])
    n_full = int(cpu_fo[-1])
    assert npu_mcp.shape[0] >= n_mask
    assert npu_fcp.shape[0] >= n_full
    assert torch.equal(npu_mcp[:n_mask].cpu().to(torch.int32), torch.from_numpy(cpu_mcp)), "mask_compact mismatch"
    assert torch.equal(npu_fcp[:n_full].cpu().to(torch.int32), torch.from_numpy(cpu_fcp)), "full_compact mismatch"

    print(
        f"[torch] shapes: mc={tuple(npu_mc.shape)} mo={tuple(npu_mo.shape)} "
        f"mcp={tuple(npu_mcp.shape)} fc={tuple(npu_fc.shape)} fo={tuple(npu_fo.shape)} "
        f"fcp={tuple(npu_fcp.shape)}"
    )
    print(f"[torch] mask_offset={npu_mo.cpu().tolist()}")
    print(f"[torch] full_offset={npu_fo.cpu().tolist()}")
    print("[torch] compact_no_sync torch example PASSED")


if __name__ == "__main__":
    test_compact_no_sync()
