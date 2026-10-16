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
N_FUNC = 2


def make_case(index):
    rng = random.Random(20260921 + index)
    batch = (2, 3, 4, 5, 8)[index % 5]
    cap = (17, 32, 65, 129, 257)[index % 5]
    lengths = [rng.randint(max(2, cap // 3), cap) for _ in range(batch)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    config = {
        "lengths": lengths,
        "offsets": offsets,
        "block_sizes": [min(length, batch + 1 + (index % 3)) for batch, length in enumerate(lengths)],
    }
    if batch > 1:
        assert len(set(config["block_sizes"])) > 1
    return config


def expected_dense(config):
    total = config["offsets"][-1]
    dense = np.zeros((total, total), dtype=np.bool_)
    for batch, length in enumerate(config["lengths"]):
        start = config["offsets"][batch]
        for query in range(length):
            for key in range(length):
                dense[start + query, start + key] = key < min(
                    (query // config["block_sizes"][batch] + 1) * config["block_sizes"][batch], length
                )
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
    tensors = {key: torch.tensor(config[key], dtype=torch.int32, device=DEVICE) for key in ["block_sizes"]}
    total = config["offsets"][-1]
    function = torch.ops.ops_rec.build_block_causal_mask_arbitrary_func
    return function(offsets, tensors["block_sizes"], total), offsets


@pytest.mark.parametrize("case_id", range(30), ids=lambda value: f"case{value + 1:02d}")
def test_build_block_causal_mask_arbitrary_func(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    expected = expected_intervals(config)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.cpu().numpy(), expected)


# ---------------- 显式边界、graph、健壮性覆盖 ----------------
_EXTRA_KEYS = ["block_sizes"]


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
        fn = torch_compiled(
            ops_rec.ascendc.attention.build_block_causal_mask_arbitrary_func, dynamic=True, fullgraph=True
        )
        return fn(offsets, *extras, total)
    return torch.ops.ops_rec.build_block_causal_mask_arbitrary_func(offsets, *extras, total)


def _compare(config, use_graph=False):
    actual = _run(config, use_graph)
    expected = expected_intervals(config)
    a = actual.cpu().numpy()
    assert a.shape == expected.shape, f"shape {a.shape} vs {expected.shape}"
    assert np.array_equal(a, expected), f"mismatch; first diffs: {np.argwhere(a != expected)[:5].tolist()}"


CASES = [
    ("single_token", [1], {"block_sizes": [1]}),
    ("block_one", [5], {"block_sizes": [1]}),
    ("block_gt_len", [4], {"block_sizes": [10]}),
    ("batch1", [7], {"block_sizes": [2]}),
    ("equal_many", [4, 4, 4], {"block_sizes": [2, 3, 4]}),
    ("mixed", [5, 9, 6], {"block_sizes": [2, 4, 1]}),
]


@pytest.mark.ci
@pytest.mark.parametrize("_name,seq_lens,kwargs", CASES)
def test_block_causal_eager(_name, seq_lens, kwargs):
    _compare(make_seq_config(seq_lens, **kwargs), use_graph=False)


@pytest.mark.graph
@pytest.mark.parametrize("_name,seq_lens,kwargs", CASES)
def test_block_causal_graph(_name, seq_lens, kwargs):
    _compare(make_seq_config(seq_lens, **kwargs), use_graph=True)


def test_block_causal_empty():
    # batch=0 时 extra tensor 必须非空且 numel==0，二者矛盾，应明确报错
    torch.npu.set_device(DEVICE)
    fn = torch.ops.ops_rec.build_block_causal_mask_arbitrary_func
    empty = torch.tensor([], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        fn(torch.tensor([0], dtype=torch.int32, device=DEVICE), empty, 0)


def test_block_causal_random_single_batch():
    rng = np.random.RandomState(17)
    for _ in range(8):
        length = int(rng.randint(1, 24))
        block = int(rng.randint(1, max(1, length)))
        _compare(make_seq_config([length], block_sizes=[block]))


def test_block_causal_large():
    _compare(make_seq_config([12345], block_sizes=[128]))
    _compare(make_seq_config([100] * 64, block_sizes=[7] * 64))


def test_block_causal_strong_checks():
    torch.npu.set_device(DEVICE)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32, device=DEVICE)
    blocks = torch.tensor([2, 2], dtype=torch.int32, device=DEVICE)
    fn = torch.ops.ops_rec.build_block_causal_mask_arbitrary_func
    with pytest.raises(RuntimeError):
        fn(offsets.unsqueeze(0), blocks, 5)
    with pytest.raises(RuntimeError):
        fn(offsets.to(torch.int64), blocks, 5)
    with pytest.raises(RuntimeError):
        fn(offsets, blocks.to(torch.int64), 5)
    with pytest.raises(RuntimeError):
        fn(offsets, blocks.unsqueeze(0), 5)
    with pytest.raises(RuntimeError):
        fn(offsets, blocks, 6)
    with pytest.raises(RuntimeError):
        fn(offsets, blocks, 4)
    with pytest.raises(RuntimeError):
        fn(offsets, blocks, -1)
    with pytest.raises(RuntimeError):
        fn(torch.tensor([0], dtype=torch.int32, device=DEVICE), blocks, 3)
    y = fn(offsets, blocks, 5)
    assert y.shape == (1, 1, 5, N_FUNC)


@pytest.mark.parametrize("case_id", [0, 4, 11, 19, 29], ids=lambda value: f"case{value + 1:02d}")
def test_block_causal_af_format(case_id):
    config = make_case(case_id)
    actual, _ = run_operator(config)
    arr = actual.cpu().numpy()
    assert arr.shape == (1, 1, arr.shape[2], N_FUNC)
    assert arr.dtype == np.int32
    for b, length in enumerate(config["lengths"]):
        start, end = config["offsets"][b], config["offsets"][b + 1]
        for query in range(start, end):
            c0, c1 = int(arr[0, 0, query, 0]), int(arr[0, 0, query, 1])
            assert start <= c0 <= c1 <= end, (b, query, c0, c1, start, end)
