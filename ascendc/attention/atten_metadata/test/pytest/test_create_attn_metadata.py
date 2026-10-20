#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""create_attn_metadata (AI CPU 自定义算子) 的 PyTorch 绑定测试。

依赖：
  1. 先编译 AICPU kernel / aclnn opapi 并安装到 vendor：
         cd cust_op/ascendc_op/ai_core_op/create_attn_metadata && bash run.sh --stage=install
     源码树直接运行时，需指向真正包含 op_impl/ 的 vendor 子目录：
         export ASCEND_CUSTOM_OPP_PATH=$PWD/build/vendor/atten_metadata
  2. 编译 PTA（common/build_ops.sh 会把 create_attn_metadata 打进 libfbgemm_npu_api.so）：
         cd cust_op/framework/torch_plugin/torch_library/common && bash build_ops.sh
"""

import os

import pytest
import torch
import ops_rec  # noqa: F401 - registers operators and packaged OPP paths

torch.npu.config.allow_internal_format = False

# metadata 布局常量，与 op_kernel_aicpu/create_attn_metadata.h 一致
METADATA_STRIDE = 16
METADATA_ALIGN = 4096
HEAD_SECTION_NUM_INDEX = 0
HEAD_IS_FD_INDEX = 1
HEAD_M_BASE_SIZE_INDEX = 2
HEAD_S2_BASE_SIZE_INDEX = 3
HEAD_AIV_CORE_NUM_INDEX = 13
HEAD_AIC_CORE_NUM_INDEX = 14
HEAD_DIRECT_MAGIC_INDEX = 15
HSTU_DIRECT_MAGIC = 0x48535455
HSTU_STREAM_K_MAGIC = 0x534B4855
FA_BN2_START_INDEX = 0
FA_M_START_INDEX = 1
FA_S2_START_INDEX = 2
FA_BN2_END_INDEX = 3
FA_M_END_INDEX = 4
FA_S2_END_INDEX = 5


def expected_size(batch, num_kv_heads, aic_core_num, aiv_core_num):
    del batch, num_kv_heads, aiv_core_num
    elems = METADATA_STRIDE + aic_core_num * METADATA_STRIDE
    return ((elems + METADATA_ALIGN - 1) // METADATA_ALIGN) * METADATA_ALIGN


def expected_dense_capacity(batch, num_kv_heads, aic_core_num, aiv_core_num):
    max_sections = max(1, batch * num_kv_heads)
    elems = METADATA_STRIDE + max_sections * (aic_core_num + aiv_core_num) * METADATA_STRIDE
    return ((elems + METADATA_ALIGN - 1) // METADATA_ALIGN) * METADATA_ALIGN


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def flatten_row_block(bn2, m_idx, row_blocks_per_batch, num_heads):
    """与 MetadataRowBlockScheduler::FlattenRowBlock 保持一致。"""
    total_bn2 = len(row_blocks_per_batch) * num_heads
    assert 0 <= bn2 <= total_bn2, f"bn2 越界: {bn2}, total={total_bn2}"
    if bn2 == total_bn2:
        assert m_idx == 0, f"末尾坐标必须是 ({total_bn2}, 0)，实际 m={m_idx}"
        return num_heads * sum(row_blocks_per_batch)

    batch_idx, head_idx = divmod(bn2, num_heads)
    row_blocks = row_blocks_per_batch[batch_idx]
    assert 0 <= m_idx <= row_blocks, f"mIdx 越界: bn2={bn2}, m={m_idx}, batch row blocks={row_blocks}"
    return num_heads * sum(row_blocks_per_batch[:batch_idx]) + head_idx * row_blocks + m_idx


def assert_all_row_blocks_covered(host_metadata, kv_seq_lens, num_heads):
    """验证 FA metadata 恰好覆盖 HSTU backward 的全部 K 行块，且没有重叠或遗漏。"""
    section_num = int(host_metadata[HEAD_SECTION_NUM_INDEX])
    is_fd = int(host_metadata[HEAD_IS_FD_INDEX])
    m_base = int(host_metadata[HEAD_M_BASE_SIZE_INDEX])
    aic_core_num = int(host_metadata[HEAD_AIC_CORE_NUM_INDEX])
    assert section_num >= 1
    assert is_fd == 0, f"HSTU metadata 不应产生 FD，实际 isFd={is_fd}"
    assert m_base > 0

    row_blocks_per_batch = [ceil_div(seq_len, m_base) for seq_len in kv_seq_lens]
    expected_block_count = num_heads * sum(row_blocks_per_batch)
    if int(host_metadata[HEAD_DIRECT_MAGIC_INDEX]) == HSTU_DIRECT_MAGIC:
        intervals = []
        for core_idx in range(aic_core_num):
            offset = METADATA_STRIDE + core_idx * METADATA_STRIDE
            begin, count = int(host_metadata[offset]), int(host_metadata[offset + 1])
            if count:
                intervals.append((begin, begin + count))
        intervals.sort()
        cursor = 0
        for begin, end in intervals:
            assert begin == cursor
            cursor = end
        assert cursor == expected_block_count
        return
    intervals = []

    for section_idx in range(section_num):
        for core_idx in range(aic_core_num):
            offset = METADATA_STRIDE + (section_idx * aic_core_num + core_idx) * METADATA_STRIDE
            record = [int(v) for v in host_metadata[offset : offset + 6]]
            bn2_start = record[FA_BN2_START_INDEX]
            m_start = record[FA_M_START_INDEX]
            bn2_end = record[FA_BN2_END_INDEX]
            m_end = record[FA_M_END_INDEX]

            start = flatten_row_block(bn2_start, m_start, row_blocks_per_batch, num_heads)
            end = flatten_row_block(bn2_end, m_end, row_blocks_per_batch, num_heads)
            assert end >= start, (
                f"FA区间倒序: section={section_idx}, core={core_idx}, record={record}, flat=[{start}, {end})"
            )
            if end == start:
                continue

            intervals.append((start, end, section_idx, core_idx))

    intervals.sort()
    covered_blocks = []
    expected_start = 0
    for start, end, section_idx, core_idx in intervals:
        assert start == expected_start, (
            f"block覆盖不连续: 期望从{expected_start}开始，实际区间[{start}, {end})，"
            f"section={section_idx}, core={core_idx}"
        )
        covered_blocks.extend(range(start, end))
        expected_start = end

    assert covered_blocks == list(range(expected_block_count)), (
        f"block覆盖错误: expected={expected_block_count}, covered={len(covered_blocks)}, intervals={intervals}"
    )


def run_create_attn_metadata(
    device,
    *,
    batch_size,
    max_seqlen_q,
    max_seqlen_kv,
    num_heads_q,
    num_heads_kv,
    head_dim,
    mask_mode,
    win_left,
    win_right,
    layout,
    seqused_q=None,
    seqused_kv=None,
    row_axis=1,
    deterministic=True,
):
    torch.npu.set_device(device)
    return torch.ops.ops_rec.create_attn_metadata(
        None,
        None,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        mask_mode,
        win_left,
        win_right,
        layout,
        layout,
        layout,
        None,
        None,
        row_axis,
        1,
        0,
        0,
        deterministic,
    )


@pytest.mark.parametrize(
    "row_axis,q_len,kv_len,block_m,block_n",
    [
        (0, 128, 32768, 128, 128),
        (1, 32768, 128, 128, 128),
    ],
)
def test_create_attn_metadata_stream_k_global_balance(row_axis, q_len, kv_len, block_m, block_n):
    """Nondeterministic metadata covers the global 2-D tile stream exactly once."""
    device = f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"
    torch.npu.set_device(device)
    metadata = torch.ops.ops_rec.create_attn_metadata(
        None,
        None,
        None,
        None,
        2,
        q_len,
        kv_len,
        4,
        4,
        128,
        0,
        -1,
        -1,
        "BSND",
        "BSND",
        "BSND",
        None,
        None,
        row_axis,
        1,
        block_m,
        block_n,
        False,
    ).cpu()
    assert int(metadata[HEAD_DIRECT_MAGIC_INDEX]) == HSTU_STREAM_K_MAGIC
    sections = int(metadata[HEAD_SECTION_NUM_INDEX])
    cores = int(metadata[HEAD_AIC_CORE_NUM_INDEX])
    outer = ceil_div(q_len if row_axis == 0 else kv_len, block_m if row_axis == 0 else block_n)
    inner = ceil_div(kv_len if row_axis == 0 else q_len, block_n if row_axis == 0 else block_m)
    expected = 2 * 4 * outer * inner
    per_core = [0] * cores
    intervals = []
    for section in range(sections):
        for core in range(cores):
            record = METADATA_STRIDE + (section * cores + core) * METADATA_STRIDE
            begin, count = int(metadata[record]), int(metadata[record + 1])
            per_core[core] += count
            if count:
                intervals.append((begin, begin + count))
    intervals.sort()
    cursor = 0
    for begin, end in intervals:
        assert begin == cursor
        cursor = end
    assert cursor == expected
    assert max(per_core) - min(per_core) <= 1


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("kv_s", [1024, 8192])
@pytest.mark.parametrize("device", [f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"])
def test_create_attn_metadata_attrs_only(batch_size, num_heads, kv_s, device):
    """仅通过属性(batch_size/max_seqlen)驱动，不传 seqused_* 张量，对应 C++ example 场景。"""
    metadata = run_create_attn_metadata(
        device,
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_kv=kv_s,
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        head_dim=128,
        mask_mode=3,
        win_left=-1,
        win_right=-1,
        layout="BSND",
    )

    assert metadata.dtype == torch.int32
    assert metadata.device.type == "npu"
    host = metadata.cpu()
    aic_core_num = int(host[HEAD_AIC_CORE_NUM_INDEX])
    aiv_core_num = int(host[HEAD_AIV_CORE_NUM_INDEX])
    assert aic_core_num > 0
    assert aiv_core_num > 0
    assert metadata.numel() == expected_dense_capacity(batch_size, num_heads, aic_core_num, aiv_core_num)
    section_num = int(host[HEAD_SECTION_NUM_INDEX])
    m_base = int(host[HEAD_M_BASE_SIZE_INDEX])
    s2_base = int(host[HEAD_S2_BASE_SIZE_INDEX])
    assert section_num >= 1, f"sectionNum 非法: {section_num}"
    assert m_base > 0, f"mBaseSize 非法: {m_base}"
    assert s2_base > 0, f"s2BaseSize 非法: {s2_base}"
    assert_all_row_blocks_covered(host, [kv_s] * batch_size, num_heads)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("device", [f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"])
def test_create_attn_metadata_with_seqused(batch_size, device):
    """通过 seqused_q / seqused_kv 张量提供实际序列长度。"""
    torch.npu.set_device(device)
    num_heads = 16
    query_seq_lens = [1, 127, 128, 385]
    kv_seq_lens = [257, 1024, 4096, 8192]
    seqused_q = torch.tensor(query_seq_lens, dtype=torch.int32, device=device)
    seqused_kv = torch.tensor(kv_seq_lens, dtype=torch.int32, device=device)

    metadata = run_create_attn_metadata(
        device,
        batch_size=-1,
        max_seqlen_q=max(query_seq_lens),
        max_seqlen_kv=max(kv_seq_lens),
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        head_dim=128,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout="BSND",
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
    )

    assert metadata.dtype == torch.int32
    host = metadata.cpu()
    aic_core_num = int(host[HEAD_AIC_CORE_NUM_INDEX])
    aiv_core_num = int(host[HEAD_AIV_CORE_NUM_INDEX])
    assert metadata.numel() >= expected_dense_capacity(batch_size, num_heads, aic_core_num, aiv_core_num)
    assert int(host[HEAD_SECTION_NUM_INDEX]) >= 1
    assert_all_row_blocks_covered(host, kv_seq_lens, num_heads)


@pytest.mark.parametrize("row_axis,expected_lengths", [(0, [1, 127, 128, 385]), (1, [257, 1024, 4096, 8192])])
def test_create_attn_metadata_direct_row_axis(row_axis, expected_lengths):
    """Direct metadata covers Q rows for forward and K rows for backward."""
    device = f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"
    query_lengths = [1, 127, 128, 385]
    kv_lengths = [257, 1024, 4096, 8192]
    metadata = run_create_attn_metadata(
        device,
        batch_size=4,
        max_seqlen_q=max(query_lengths),
        max_seqlen_kv=max(kv_lengths),
        num_heads_q=4,
        num_heads_kv=4,
        head_dim=64,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout="BSND",
        seqused_q=torch.tensor(query_lengths, dtype=torch.int32, device=device),
        seqused_kv=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        row_axis=row_axis,
    ).cpu()
    assert_all_row_blocks_covered(metadata, expected_lengths, 4)


def test_create_attn_metadata_packed_direct_range():
    """Packed counts use weighted direct-range scheduling."""
    device = f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"
    torch.npu.set_device(device)
    batch_size, num_heads, max_seq = 3, 4, 1024
    seq_lens = [257, 513, 1024]
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    outer = ceil_div(max_seq, 128)
    mask_cnt = torch.zeros((batch_size, 1, outer), dtype=torch.int32, device=device)
    full_cnt = torch.zeros_like(mask_cnt)
    for batch, length in enumerate(seq_lens):
        rows = ceil_div(length, 128)
        mask_cnt[batch, 0, :rows] = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        full_cnt[batch, 0, :rows] = 1

    metadata = torch.ops.ops_rec.create_attn_metadata(
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        batch_size,
        max_seq,
        max_seq,
        num_heads,
        num_heads,
        64,
        0,
        -1,
        -1,
        "TND",
        "TND",
        "TND",
        mask_cnt,
        full_cnt,
        1,
        1,
    ).cpu()
    assert int(metadata[15]) == HSTU_DIRECT_MAGIC
    assert int(metadata[HEAD_SECTION_NUM_INDEX]) == 1
    assert metadata.numel() == expected_size(
        batch_size, 1, int(metadata[HEAD_AIC_CORE_NUM_INDEX]), int(metadata[HEAD_AIV_CORE_NUM_INDEX])
    )
    assert_all_row_blocks_covered(metadata, seq_lens, num_heads)


def test_create_attn_metadata_packed_heavy_row_uses_nearest_boundary():
    """A heavy row goes to the side closest to the cumulative target."""
    device = f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"
    torch.npu.set_device(device)
    batch_size, num_heads, max_seq = 1, 1, 4096
    cu_seqlens = torch.tensor([0, max_seq], dtype=torch.int32, device=device)
    rows = ceil_div(max_seq, 128)
    mask_cnt = torch.ones((batch_size, 1, rows), dtype=torch.int32, device=device)
    full_cnt = torch.zeros_like(mask_cnt)
    mask_cnt[0, 0, 5] = 96

    metadata = torch.ops.ops_rec.create_attn_metadata(
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        batch_size,
        max_seq,
        max_seq,
        num_heads,
        num_heads,
        64,
        0,
        -1,
        -1,
        "TND",
        "TND",
        "TND",
        mask_cnt,
        full_cnt,
        1,
        1,
    ).cpu()
    assert_all_row_blocks_covered(metadata, [max_seq], num_heads)

    core_num = int(metadata[HEAD_AIC_CORE_NUM_INDEX])
    counts = []
    begins = []
    for core in range(core_num):
        record = METADATA_STRIDE + core * METADATA_STRIDE
        begins.append(int(metadata[record]))
        counts.append(int(metadata[record + 1]))
    weights = mask_cnt.cpu().flatten().tolist()
    loads = [sum(weights[begin : begin + count]) for begin, count in zip(begins, counts)]
    # The old overshoot-only split assigns rows [0..5] to core 0 (load 101).
    # Nearest-boundary splitting stops before the heavy row (load 5).
    assert begins[0] == 0
    assert counts[0] == 5
    assert loads[0] == 5
    assert all(count > 0 for count in counts)


def test_create_attn_metadata_packed_multi_batch_keeps_multiple_cores():
    """The B>1 fallback must retain the established multi-core split."""
    device = f"npu:{os.getenv('ASCEND_DEVICE_ID', '0')}"
    torch.npu.set_device(device)
    batch_size, num_heads, max_seq = 8, 4, 1024
    seq_lens = [1024, 896, 768, 640, 512, 384, 256, 128]
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    outer = ceil_div(max_seq, 128)
    mask_cnt = torch.ones((batch_size, 1, outer), dtype=torch.int32, device=device)
    full_cnt = torch.ones_like(mask_cnt)

    metadata = torch.ops.ops_rec.create_attn_metadata(
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        batch_size,
        max_seq,
        max_seq,
        num_heads,
        num_heads,
        64,
        0,
        -1,
        -1,
        "TND",
        "TND",
        "TND",
        mask_cnt,
        full_cnt,
        1,
        1,
    ).cpu()
    assert_all_row_blocks_covered(metadata, seq_lens, num_heads)
    core_num = int(metadata[HEAD_AIC_CORE_NUM_INDEX])
    counts = [int(metadata[METADATA_STRIDE + core * METADATA_STRIDE + 1]) for core in range(core_num)]
    assert sum(count > 0 for count in counts) > 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
