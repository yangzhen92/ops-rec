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
"""Invalid-argument tests for create_q2k/k2q_block_sparse_from_func.

Split out of ``test_create_block_mask.py`` (which exceeded 1000 lines);
positive cases live there, here we cover argument validation plus the
Q2K kv_split_mode (multi-warp KV split) correctness sweep (section 10).
"""

import re
import pytest
import torch

import ops_rec

from test_create_block_mask import _check_k2q, _check_q2k, generate_func_tensor, generate_special_func_tensor

CPU_DEVICE = "cpu"
NPU_DEVICE = "npu:0"


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU device not available")


def _to_npu(func_tensor_cpu):
    return func_tensor_cpu.to(NPU_DEVICE)


# ---------------------------------------------------------------------------
# 7. Block-size alignment argument validation
#
# Both the Q2K and K2Q entry points in block_sparse_func_plugin.cpp enforce
# (catlass restriction):
#
#     TORCH_CHECK(q_block_size > 0 && q_block_size % 32 == 0,
#                 "q_block_size must be positive and divisible by 32");
#     TORCH_CHECK(kv_block_size > 0 && kv_block_size % 32 == 0,
#                 "kv_block_size must be positive and divisible by 32");
#
# The tests below verify that a non-aligned (or non-positive) block size is
# rejected by both Q2K and K2Q with the matching message (legal block sizes
# are covered by test_different_block_sizes above).
# ---------------------------------------------------------------------------
Q_BLOCK_SIZE_ERR_MSG = "q_block_size must be positive and divisible by 32"
KV_BLOCK_SIZE_ERR_MSG = "kv_block_size must be positive and divisible by 32"

# 非 32 的倍数或非正数——每一个取值都必须被拒绝
INVALID_BLOCK_SIZES = [
    31,  # 最小对齐尺寸减一
    33,  # 32 加一
    48,  # 非 32 的倍数
    100,  # 非 32 的倍数
    0,  # 非正数
    -32,  # 负数（虽为 32 的倍数）
    -1,  # 负数
]


def _minimal_npu_func_tensor(seqlen_q=64, n_func=2, B=1, H=1):
    """构造最小规模的 NPU 布局 func_tensor [B, H, func_q_len, n_func]（n_func 为偶数）。"""
    func_tensor = torch.zeros(B, H, seqlen_q, n_func, dtype=torch.int32, device=CPU_DEVICE)
    return _to_npu(func_tensor.contiguous())


@pytest.mark.parametrize("bad_q_block_size", INVALID_BLOCK_SIZES)
def test_invalid_q_block_size(bad_q_block_size):
    """Q2K and K2Q both reject a q_block_size that is not a positive multiple of 32."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor()
    # q2k: 非法 q_block_size（合法 kv_block_size）
    with pytest.raises(RuntimeError, match=Q_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=bad_q_block_size,
            kv_block_size=128,
            check_q_boundary=True,
            debug=False,
        )
    # k2q: 非法 q_block_size（合法 kv_block_size）
    with pytest.raises(RuntimeError, match=Q_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=bad_q_block_size,
            kv_block_size=128,
            debug=False,
        )


@pytest.mark.parametrize("bad_kv_block_size", INVALID_BLOCK_SIZES)
def test_invalid_kv_block_size(bad_kv_block_size):
    """Q2K and K2Q both reject a kv_block_size that is not a positive multiple of 32."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor()
    # q2k: 合法 q_block_size，非法 kv_block_size
    with pytest.raises(RuntimeError, match=KV_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=bad_kv_block_size,
            check_q_boundary=True,
            debug=False,
        )
    # k2q: 合法 q_block_size，非法 kv_block_size
    with pytest.raises(RuntimeError, match=KV_BLOCK_SIZE_ERR_MSG):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=bad_kv_block_size,
            debug=False,
        )


# ---------------------------------------------------------------------------
# 8. func_q_len / q_len consistency validation
#
# Both the Q2K and K2Q entry points enforce:
#
#     TORCH_CHECK(func_q_len >= q_len,
#                 "func_q_len must be greater than or equal to q_len");
#
# The func_tensor must cover the whole q sequence; a func_q_len smaller than
# q_len is rejected (func_q_len == q_len is legal and covered by the positive
# tests in test_create_block_mask.py).
# ---------------------------------------------------------------------------
FUNC_Q_LEN_ERR_MSG = "func_q_len must be greater than or equal to q_len"

# func_q_len < q_len 的每一种取值都必须被拒绝（q_len 固定为 64）
INVALID_FUNC_Q_LENS = [
    63,  # q_len - 1
    32,  # 恰为一半
    1,  # 最小正数
]


@pytest.mark.parametrize("bad_func_q_len", INVALID_FUNC_Q_LENS)
def test_invalid_func_q_len_less_than_q_len(bad_func_q_len):
    """Q2K and K2Q both reject a func_q_len smaller than q_len."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor(seqlen_q=bad_func_q_len)
    # q2k
    with pytest.raises(RuntimeError, match=FUNC_Q_LEN_ERR_MSG):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=128,
            check_q_boundary=True,
            debug=False,
        )
    # k2q
    with pytest.raises(RuntimeError, match=FUNC_Q_LEN_ERR_MSG):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=128,
            debug=False,
        )


# ---------------------------------------------------------------------------
# 9. n_func parity validation
#
# Both the Q2K and K2Q entry points enforce:
#
#     TORCH_CHECK(n_func % 2 == 0,
#                 "n_func must be even (e.g., 2, 4, 6, 8, ...)");
#
# n_func encodes half-open interval pairs [F0, F1) ∪ [F2, F3) ∪ ... so it must
# be even; every odd value must be rejected (legal even values are covered by
# the positive tests in test_create_block_mask.py).
# ---------------------------------------------------------------------------
N_FUNC_ERR_MSG = "n_func must be even (e.g., 2, 4, 6, 8, ...)"
# 报错文案含括号，match 为正则，需转义
N_FUNC_ERR_PATTERN = re.escape(N_FUNC_ERR_MSG)

# 奇数 n_func 的每一种取值都必须被拒绝
INVALID_N_FUNCS = [
    1,  # 最小奇数
    3,
    5,
    7,
]


@pytest.mark.parametrize("bad_n_func", INVALID_N_FUNCS)
def test_invalid_odd_n_func(bad_n_func):
    """Q2K and K2Q both reject an odd n_func."""
    _require_npu()
    func_tensor = _minimal_npu_func_tensor(seqlen_q=64, n_func=bad_n_func)
    # q2k
    with pytest.raises(RuntimeError, match=N_FUNC_ERR_PATTERN):
        ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=128,
            check_q_boundary=True,
            debug=False,
        )
    # k2q
    with pytest.raises(RuntimeError, match=N_FUNC_ERR_PATTERN):
        ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor=func_tensor,
            q_len=64,
            kv_len=128,
            q_block_size=128,
            kv_block_size=128,
            debug=False,
        )


@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [(128, 128), (512, 128), (512, 256), (256, 512)])
def test_different_block_sizes_k2q(Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Q and KV block sizes can be chosen independently."""
    _require_npu()
    seqlen = 1024
    func_tensor_cpu = generate_special_func_tensor(
        seqlen,
        seqlen,
        "causal",
        n_func=3,
    )
    _check_k2q(
        func_tensor_cpu,
        seqlen,
        seqlen,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        tag=f"block_sizes(Q_BS={Q_BLOCK_SIZE}, KV_BS={KV_BLOCK_SIZE})",
    )


# ---------------------------------------------------------------------------
# 10. kv_split_mode (multi-warp KV split) correctness — Q2K only
#
# op_host/create_q2k_block_sparse_from_func.cpp (under ENABLE_WARP_LEVEL_OPT,
# unconditionally defined in block_sparse_common.h) switches the Q2K kernel to
# the multi-warp KV split when ALL of the following hold:
#
#     use_warp_level = (q_block_size <= 256) && (q_block_size % 32 == 0)
#     split_waves    = DIVUP(totalBlocks, aivCoreCount) <= 2
#     kv_split_mode  = use_warp_level && 16 <= numKvBlocks <= 2048
#
# where totalBlocks = numQBlocks * H * B and 2048 = SPLIT_MAX_KV_BLOCKS
# (8 warps * 256 staged kv blocks per warp). In split mode one 256-thread
# block owns a single q_block, its 8 warps scan disjoint 1/8 kv sub-ranges,
# stage the full/mask ids in UB and write them back with warp-prefix offsets
# (full ascending at the row front, mask descending at the row back — the
# same ordering as the non-split path). K2Q has no kv_split_mode.
#
# The mode flag only lives in device-side tiling data and cannot be observed
# from Python, so these tests pin the shape parameters to combinations that
# activate the mode (or stay just outside its window) and verify the Q2K
# outputs against the CPU reference. All mode-on cases keep totalBlocks <= 16
# so split_waves == 1 on any current AIV part (>= 40 AIV cores).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pattern",
    [
        "diagonal",  # 每 q_block 仅覆盖 1 个 kv_block：span=1，7 个 warp 空转（余数分支）
        "anti_causal",  # span 从 numKvBlocks 递减到 1：满跨度 + 非整除余数
    ],
)
def test_kv_split_mode_special_patterns(pattern):
    """Q2K results stay correct in kv_split_mode for small-span and full-span masks."""
    _require_npu()
    q_len, kv_len = 512, 4096
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    # numQBlocks=4, B=H=1 -> totalBlocks=4（split_waves=1）；numKvBlocks=32 -> 模式开启
    func_tensor_cpu = generate_special_func_tensor(q_len, kv_len, pattern, n_func=3, block_size=KV_BLOCK_SIZE)
    _check_q2k(
        func_tensor_cpu,
        q_len,
        kv_len,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"kv_split_mode(pattern={pattern})",
    )


def test_kv_split_mode_random():
    """Random masks scatter full/mask ids across all 8 warps (prefix write-out)."""
    _require_npu()
    q_len, kv_len = 1024, 8192
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    # numQBlocks=8 -> totalBlocks=8；numKvBlocks=64 -> 模式开启
    func_tensor_cpu = generate_func_tensor(q_len, kv_len, 3, seed=q_len * 100 + Q_BLOCK_SIZE)
    _check_q2k(
        func_tensor_cpu,
        q_len,
        kv_len,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag="kv_split_mode_random(n_func=3)",
    )


@pytest.mark.parametrize("Q_BLOCK_SIZE", [32, 256])
def test_kv_split_mode_q_block_size(Q_BLOCK_SIZE):
    """Warp-level-eligible q block sizes (32 and 256) both run the split path."""
    _require_npu()
    q_len, kv_len = 512, 2048
    KV_BLOCK_SIZE = 128
    num_kv_blocks = (kv_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    assert num_kv_blocks == 16  # 恰在下界，模式开启
    # q_block_size=32 时 numQBlocks=16 -> totalBlocks=16，split_waves 仍为 1
    func_tensor_cpu = generate_special_func_tensor(q_len, kv_len, "anti_causal", n_func=3, block_size=KV_BLOCK_SIZE)
    _check_q2k(
        func_tensor_cpu,
        q_len,
        kv_len,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"kv_split_mode_q_block_size(Q_BS={Q_BLOCK_SIZE})",
    )


@pytest.mark.parametrize(
    "kv_len,mode_expected",
    [
        (1920, "off"),  # numKvBlocks = 15 < 16：下界之外，走普通 warp 级路径
        (2048, "on"),  # numKvBlocks = 16：下界
        (262144, "on"),  # numKvBlocks = 2048：上界，每 warp 恰暂存 256 个 kv_block
    ],
)
def test_kv_split_mode_num_kv_blocks_boundary(kv_len, mode_expected):
    """Correctness on both sides of the [16, 2048] numKvBlocks activation window."""
    _require_npu()
    q_len = 256
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    num_kv_blocks = (kv_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    if mode_expected == "on":
        assert 16 <= num_kv_blocks <= 2048
    else:
        assert num_kv_blocks < 16 or num_kv_blocks > 2048
    func_tensor_cpu = generate_special_func_tensor(q_len, kv_len, "anti_causal", n_func=3, block_size=KV_BLOCK_SIZE)
    _check_q2k(
        func_tensor_cpu,
        q_len,
        kv_len,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"kv_split_mode_kv_boundary(kv_len={kv_len}, numKvBlocks={num_kv_blocks}, mode={mode_expected})",
    )


def test_kv_split_mode_partial_q_block():
    """A partial last q_block (check_q_boundary=True) can never be FULL in split mode."""
    _require_npu()
    q_len, kv_len = 500, 4096
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    # q_len=500 -> numQBlocks=4，最后一个 q_block 仅 116/128 个 token
    func_tensor_cpu = generate_special_func_tensor(q_len, kv_len, "sliding_window", n_func=3, block_size=KV_BLOCK_SIZE)
    _check_q2k(
        func_tensor_cpu,
        q_len,
        kv_len,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        check_q_boundary=True,
        tag=f"kv_split_mode_partial_q(q_len={q_len})",
    )


# ---------------------------------------------------------------------------
# 3b. K2Q "many small kv blocks": q_len small, kv_len large, kv_block_size small
#     (num_kv_blocks >= 100+) — exercises the single-phase warp-per-column path
#     with dynamic kvColsPerTask.
# ---------------------------------------------------------------------------
@pytest.mark.ci
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (2048, 32768),  # nq=16, nk=1024
        (1024, 16384),  # nq=8, nk=512
        (4999, 33333),  # non-aligned: nq=40, nk=1042
    ],
)
@pytest.mark.parametrize("n_func", [3, 5])
def test_k2q_many_small_kv_blocks(seqlen_q, seqlen_k, n_func):
    """K2Q with tiny kv_block_size (32) producing many kv blocks."""
    _require_npu()
    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 32
    seed = seqlen_q * 100 + n_func * 10 + KV_BLOCK_SIZE
    func_tensor_cpu = generate_func_tensor(seqlen_q, seqlen_k, n_func, seed=seed)
    tag = f"k2q_many_small_kv(seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, n_func={n_func}, KV_BS={KV_BLOCK_SIZE})"
    _check_k2q(func_tensor_cpu, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, tag=tag)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
