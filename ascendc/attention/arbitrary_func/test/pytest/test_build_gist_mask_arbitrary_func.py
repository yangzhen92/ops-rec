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

import random
import numpy as np
import pytest
import torch
import ops_rec  # noqa: F401
import sys

torch.npu.config.allow_internal_format = False

# dynamic=True 图模式会在不同 shape 下重编译，增大 cache 避免超限报错
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


DEVICE = "npu:0"
N_FUNC = 4


def make_case(index):
    rng = random.Random(20260921 + 5000 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (33, 65, 129, 257, 513)[index % 5]
    lengths = [rng.randint(max(16, cap // 2), cap) for _ in range(batch)]
    lengths[0] = cap
    offsets = [0]
    pre_doc_lens = []
    gist_starts = []
    gist_ends = []
    for batch_id, length in enumerate(lengths):
        pre_doc = min(3 + (index + batch_id) % 7, max(1, length // 4))
        gist_len = min(4 + (index + 2 * batch_id) % 11, max(1, length // 5))
        doc_min = pre_doc + 1
        gist_start = rng.randint(doc_min, max(doc_min, length - gist_len - 1))
        gist_end = min(length, gist_start + gist_len)
        if gist_end >= length:
            gist_end = length - 1
            gist_start = max(pre_doc + 1, gist_end - gist_len)
        pre_doc_lens.append(pre_doc)
        gist_starts.append(gist_start)
        gist_ends.append(gist_end)
        offsets.append(offsets[-1] + length)
    return {
        "lengths": lengths,
        "offsets": offsets,
        "pre_doc_lens": pre_doc_lens,
        "gist_starts": gist_starts,
        "gist_ends": gist_ends,
    }


def add_interval(intervals, begin, end, seq_start, seq_end):
    begin = max(begin, seq_start)
    end = min(end, seq_end)
    if begin >= end:
        return
    if intervals and intervals[-1][1] == begin:
        intervals[-1] = (intervals[-1][0], end)
    else:
        intervals.append((begin, end))


def expected_intervals(config):
    total = config["offsets"][-1]
    encoded = np.zeros((1, 1, total, N_FUNC), dtype=np.int32)
    for batch_id, length in enumerate(config["lengths"]):
        seq_start, seq_end = config["offsets"][batch_id : batch_id + 2]
        p = config["pre_doc_lens"][batch_id]
        g0 = config["gist_starts"][batch_id]
        g1 = config["gist_ends"][batch_id]
        for local_q in range(length):
            global_q = seq_start + local_q
            intervals = []
            if local_q < p:
                add_interval(intervals, seq_start, global_q + 1, seq_start, seq_end)
            elif local_q < g0:
                add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
                add_interval(intervals, seq_start + p, global_q + 1, seq_start, seq_end)
            elif local_q < g1:
                add_interval(intervals, seq_start, global_q + 1, seq_start, seq_end)
            else:
                add_interval(intervals, seq_start, seq_start + p, seq_start, seq_end)
                add_interval(intervals, seq_start + g0, seq_start + g1, seq_start, seq_end)
                add_interval(intervals, seq_start + g1, global_q + 1, seq_start, seq_end)
            while len(intervals) < 2:
                last = intervals[-1][1] if intervals else seq_start
                intervals.append((last, last))
            assert len(intervals) <= 2
            encoded[0, 0, global_q] = np.asarray(intervals[:2], dtype=np.int32).reshape(-1)
    return encoded


def expected_dense(config):
    encoded = expected_intervals(config)
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for row in range(total):
        for begin, end in encoded[0, 0, row].reshape(-1, 2):
            if begin < end:
                dense[row, begin:end] = True
    return dense


def run_operator(config):
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)
    pre_doc_lens = torch.tensor(config["pre_doc_lens"], dtype=torch.int32, device=DEVICE)
    gist_starts = torch.tensor(config["gist_starts"], dtype=torch.int32, device=DEVICE)
    gist_ends = torch.tensor(config["gist_ends"], dtype=torch.int32, device=DEVICE)
    total = config["offsets"][-1]
    return torch.ops.ops_rec.build_gist_mask_arbitrary_func(
        offsets, pre_doc_lens, gist_starts, gist_ends, total
    ), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_gist_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


# ---------------- 显式边界、graph、健壮性覆盖----------------
_EXTRA_KEYS = ["pre_doc_lens", "gist_starts", "gist_ends"]


def make_seq_config(seq_lens, **params):
    lengths = list(seq_lens)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    config = {"lengths": lengths, "offsets": offsets}
    config.update(params)
    return config


def _run(config, use_graph=False):
    offsets = torch.tensor(config["offsets"], dtype=torch.int32, device=DEVICE)
    extras = [torch.tensor(config[key], dtype=torch.int32, device=DEVICE) for key in _EXTRA_KEYS]
    total = config["offsets"][-1]
    if use_graph:
        fn = torch_compiled(ops_rec.ascendc.attention.build_gist_mask_arbitrary_func, dynamic=True, fullgraph=True)
        return fn(offsets, *extras, total)
    return torch.ops.ops_rec.build_gist_mask_arbitrary_func(offsets, *extras, total)


def _compare(config, use_graph=False):
    actual = _run(config, use_graph)
    expected = expected_intervals(config)
    a = actual.cpu().numpy()
    assert a.shape == expected.shape, f"shape {a.shape} vs {expected.shape}"
    assert np.array_equal(a, expected), f"mismatch; first diffs: {np.argwhere(a != expected)[:5].tolist()}"


CASES = [
    ("single_token", [1], {"pre_doc_lens": [1], "gist_starts": [1], "gist_ends": [1]}),
    ("no_gist", [6], {"pre_doc_lens": [2], "gist_starts": [6], "gist_ends": [6]}),
    ("no_predoc", [6], {"pre_doc_lens": [0], "gist_starts": [2], "gist_ends": [4]}),
    ("brief", [4], {"pre_doc_lens": [1], "gist_starts": [2], "gist_ends": [3]}),
    ("query_after_gist", [8], {"pre_doc_lens": [2], "gist_starts": [3], "gist_ends": [5]}),
    ("batch1", [7], {"pre_doc_lens": [2], "gist_starts": [3], "gist_ends": [5]}),
    ("equal_many", [5, 5, 5], {"pre_doc_lens": [1, 2, 4], "gist_starts": [2, 3, 5], "gist_ends": [3, 4, 5]}),
    ("mixed", [8, 6, 9], {"pre_doc_lens": [2, 0, 3], "gist_starts": [3, 4, 6], "gist_ends": [5, 5, 7]}),
]


@pytest.mark.ci
@pytest.mark.parametrize("_name,seq_lens,kwargs", CASES)
def test_gist_eager(_name, seq_lens, kwargs):
    _compare(make_seq_config(seq_lens, **kwargs), use_graph=False)


@pytest.mark.graph
@pytest.mark.parametrize("_name,seq_lens,kwargs", CASES)
def test_gist_graph(_name, seq_lens, kwargs):
    _compare(make_seq_config(seq_lens, **kwargs), use_graph=True)


def test_gist_empty():
    # batch=0 时 extra tensor 必须非空且 numel==0，二者矛盾，应明确报错
    torch.npu.set_device(DEVICE)
    fn = torch.ops.ops_rec.build_gist_mask_arbitrary_func
    empty = torch.tensor([], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        fn(torch.tensor([0], dtype=torch.int32, device=DEVICE), empty, empty, empty, 0)


def test_gist_random_single_batch():
    rng = np.random.RandomState(43)
    for _ in range(8):
        length = int(rng.randint(4, 24))
        p = int(rng.randint(0, length // 2))
        g0 = int(rng.randint(p, max(p, length - 2)))
        g1 = int(rng.randint(g0, max(g0, length - 1)))
        _compare(make_seq_config([length], pre_doc_lens=[p], gist_starts=[g0], gist_ends=[g1]))


def test_gist_large():
    _compare(make_seq_config([12345], pre_doc_lens=[2000], gist_starts=[4000], gist_ends=[6000]))
    _compare(make_seq_config([100] * 64, pre_doc_lens=[5] * 64, gist_starts=[13] * 64, gist_ends=[29] * 64))


def test_gist_strong_checks():
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)
    predoc = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    gstarts = torch.tensor([2, 2], dtype=torch.int32, device=DEVICE)
    gends = torch.tensor([3, 2], dtype=torch.int32, device=DEVICE)
    fn = torch.ops.ops_rec.build_gist_mask_arbitrary_func
    with pytest.raises(RuntimeError):
        fn(offsets.unsqueeze(0), predoc, gstarts, gends, 5)
    with pytest.raises(RuntimeError):
        fn(offsets.to(torch.int64), predoc, gstarts, gends, 5)
    with pytest.raises(RuntimeError):
        fn(offsets, predoc.to(torch.int64), gstarts, gends, 5)
    with pytest.raises(RuntimeError):
        fn(offsets, predoc, gstarts.unsqueeze(0), gends, 5)
    with pytest.raises(RuntimeError):
        fn(offsets, predoc, gstarts, gends, 6)
    with pytest.raises(RuntimeError):
        fn(offsets, predoc, gstarts, gends, 4)
    with pytest.raises(RuntimeError):
        fn(offsets, predoc, gstarts, gends, -1)
    with pytest.raises(RuntimeError):
        fn(torch.tensor([0], dtype=torch.int32, device=DEVICE), predoc, gstarts, gends, 3)
    y = fn(offsets, predoc, gstarts, gends, 5)
    assert y.shape == (1, 1, 5, N_FUNC)


@pytest.mark.parametrize("case_id", [0, 4, 11, 19, 29], ids=lambda value: f"case{value + 1:02d}")
def test_gist_af_format(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    arr = actual.cpu().numpy()
    assert arr.shape == (1, 1, arr.shape[2], N_FUNC)
    assert arr.dtype == np.int32
    for b, length in enumerate(config["lengths"]):
        start, end = config["offsets"][b], config["offsets"][b + 1]
        for query in range(start, end):
            c0, c1 = int(arr[0, 0, query, 0]), int(arr[0, 0, query, 1])
            c2, c3 = int(arr[0, 0, query, 2]), int(arr[0, 0, query, 3])
            assert start <= c0 <= c1 <= end, (b, query, c0, c1, start, end)
            assert start <= c2 <= c3 <= end, (b, query, c2, c3, start, end)
