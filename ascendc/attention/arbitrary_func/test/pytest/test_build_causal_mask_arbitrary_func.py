"""Independent correctness, boundary, graph and af-format tests for causal.

Causal semantics: each query q in a sequence sees every key k with k <= q
(single interval [seq_start, q+1), n_func=2).
"""

import random
import sys

import numpy as np
import pytest
import torch
import ops_rec  # noqa: F401

DEVICE = "npu:0"
N_FUNC = 2

torch.npu.config.allow_internal_format = False

# dynamic=True 图模式会在不同 shape 下重编译，增大 cache 避免超限报错
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def build_causal_mask_arbitrary_func_cpu(offsets, total_seqlen):
    """NumPy reference (single interval, n_func=2), output [1, 1, total, 2]."""
    offsets = np.asarray(offsets, dtype=np.int32)
    af = np.zeros((1, 1, total_seqlen, N_FUNC), dtype=np.int32)
    positions = np.arange(total_seqlen, dtype=np.int32)
    seq_id = np.searchsorted(offsets[1:] - 1, positions, side="left")
    af[0, 0, :, 0] = offsets[seq_id]      # seq_start
    af[0, 0, :, 1] = positions + 1        # global pos + 1（含自身）
    return af


def make_sequences(seq_lens):
    """jagged 序列长度列表 -> offsets / total_seqlen"""
    offsets = np.zeros(len(seq_lens) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(seq_lens, dtype=np.int32)
    return offsets, int(offsets[-1])


def run_compare(offsets, total_seqlen, use_graph=False):
    torch.npu.set_device(DEVICE)
    cpu = build_causal_mask_arbitrary_func_cpu(offsets, total_seqlen)
    offsets_t = torch.from_numpy(np.asarray(offsets, dtype=np.int32)).to(DEVICE)

    if use_graph:
        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.build_causal_mask_arbitrary_func, dynamic=True, fullgraph=True
        )
        y = compiled_fn(offsets_t, total_seqlen)
    else:
        y = ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(offsets_t, total_seqlen)

    y_np = y.cpu().numpy()
    assert y_np.shape == (1, 1, total_seqlen, N_FUNC), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), f"mismatch; first diffs: {np.argwhere(y_np != cpu)[:5].tolist()}"


# (name, seq_lens)：覆盖单 token、batch=1、等长多序列、混合边界等
CASES = [
    ("single_token", [1]),
    ("single_token_multi", [1, 1, 1]),
    ("batch1", [7]),
    ("two_equal", [3, 3]),
    ("equal_many", [4, 4, 4, 4]),
    ("mixed_boundary", [1, 3, 7, 2]),
    ("one_short_one_long", [1, 21]),
]


@pytest.mark.ci
@pytest.mark.parametrize("name,seq_lens", CASES)
def test_build_causal_mask_arbitrary_func_eager(name, seq_lens):
    offsets, total = make_sequences(seq_lens)
    run_compare(offsets, total, use_graph=False)


@pytest.mark.graph
@pytest.mark.parametrize("name,seq_lens", CASES)
def test_build_causal_mask_arbitrary_func_graph(name, seq_lens):
    offsets, total = make_sequences(seq_lens)
    run_compare(offsets, total, use_graph=True)


def test_build_causal_mask_arbitrary_func_empty():
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    y = ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(offsets, 0)
    assert y.shape == (1, 1, 0, N_FUNC)
    assert y.numel() == 0


def test_build_causal_mask_arbitrary_func_random_single_batch():
    rng = np.random.RandomState(7)
    for _ in range(10):
        length = int(rng.randint(1, 32))
        offsets, total = make_sequences([length])
        run_compare(offsets, total, use_graph=False)


def test_build_causal_mask_arbitrary_func_large():
    # 单序列大 shape，跨多核遍历
    run_compare(np.array([0, 12345], dtype=np.int32), 12345)
    # 多 batch 大 shape
    offsets, total = make_sequences([100] * 64)
    run_compare(offsets, total)


def test_build_causal_mask_arbitrary_func_strong_checks():
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)  # [bs+1], bs=2
    fn = ops_rec.ascendc.attention.build_causal_mask_arbitrary_func

    # offsets 非 1D
    with pytest.raises(RuntimeError):
        fn(offsets.unsqueeze(0), 5)
    # offsets 数据类型非 int32
    with pytest.raises(RuntimeError):
        fn(offsets.to(torch.int64), 5)
    # total_seqlen 与 offsets[-1] 不一致
    with pytest.raises(RuntimeError):
        fn(offsets, 6)
    with pytest.raises(RuntimeError):
        fn(offsets, 4)
    # total_seqlen 为负
    with pytest.raises(RuntimeError):
        fn(offsets, -1)
    # offsets 为空
    with pytest.raises(RuntimeError):
        fn(torch.tensor([], dtype=torch.int32, device=DEVICE), 0)

    # 合法输入不报错
    y = fn(offsets, 5)
    assert y.shape == (1, 1, 5, N_FUNC)


@pytest.mark.parametrize("case_id", [0, 4, 11, 19, 29], ids=lambda value: f"case{value + 1:02d}")
def test_causal_af_format(case_id):
    """独立校验 af 输出符合统一格式约定：2 列、半开区间、不越出所属序列。"""
    config = make_case(case_id)
    actual, _ = run_operator(config)
    arr = actual.cpu().numpy()
    total = config["offsets"][-1]
    assert arr.shape == (1, 1, total, N_FUNC)
    assert arr.dtype == np.int32
    for batch, length in enumerate(config["lengths"]):
        start, end = config["offsets"][batch], config["offsets"][batch + 1]
        for query in range(start, end):
            c0, c1 = int(arr[0, 0, query, 0]), int(arr[0, 0, query, 1])
            # 半开区间合法（空区间 c0 == c1），且不越出所属序列
            assert 0 <= c0 <= c1, (batch, query, c0, c1)
            assert start <= c0 and c1 <= end, (batch, query, c0, c1, start, end)


# ---------------- 原随机正确性测试（保留） ----------------

def make_case(index):
    rng = random.Random(20260921 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (17, 32, 65, 129, 257)[index % 5]
    lengths = [rng.randint(max(2, cap // 3), cap) for _ in range(batch)]
    lengths[0] = cap
    lengths[-1] = max(2, cap // 2)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    config = {
        "lengths": lengths,
        "offsets": offsets,
    }
    return config


def expected_dense(config):
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for batch, length in enumerate(config["lengths"]):
        start = config["offsets"][batch]
        for query in range(length):
            for key in range(length):
                dense[start + query, start + key] = key <= query
    return dense


def expected_intervals(config):
    dense = expected_dense(config)
    total = dense.shape[0]
    encoded = np.zeros((1, 1, total, N_FUNC), dtype=np.int32)
    for batch, length in enumerate(config["lengths"]):
        start, end = config["offsets"][batch : batch + 2]
        for query in range(start, end):
            active = np.flatnonzero(dense[query, start:end]) + start
            intervals = []
            if active.size:
                run_start = run_end = int(active[0])
                for key in active[1:]:
                    key = int(key)
                    if key != run_end + 1:
                        intervals.append((run_start, run_end + 1))
                        run_start = key
                    run_end = key
                intervals.append((run_start, run_end + 1))
            if not intervals:
                intervals.append((start, start))
            assert len(intervals) == 1
            encoded[0, 0, query] = np.asarray(intervals, dtype=np.int32).reshape(-1)
    return encoded


def run_operator(config):
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)

    total = config["offsets"][-1]
    function = torch.ops.ops_rec.build_causal_mask_arbitrary_func
    return function(offsets, total), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_causal_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
