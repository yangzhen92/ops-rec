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

import sys
import pytest
import torch
import numpy as np

import ops_rec

DEVICE = "npu:0"

torch.npu.config.allow_internal_format = False

# dynamic=True 图模式会在不同 shape 下重编译，增大 cache 避免超限报错
torch._dynamo.config.cache_size_limit = 256


def torch_compiled(model, **kwargs):
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    return model


def build_sla_mask_arbitrary_func_cpu(offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets=None):
    """
    history token: c0=min(pos+1,effK2) / c1=max(effK2,pos-K1+1) / c2=pos+1
    target token : c0=c1=c2=H（只见 [0,H)，连自身不可见）
    af[q] = [S, S+c0, S+c1, S+c2]（全局坐标）
    """
    batch_size = len(offsets) - 1
    af = np.zeros((1, 1, total_seqlen, 4), dtype=np.int32)
    eff_k2 = max(sla_k2, contextual_seq_len)
    if num_targets is None:
        num_targets = np.zeros(batch_size, dtype=np.int32)
    for pos in range(total_seqlen):
        lo, hi = 0, batch_size
        while lo < hi:
            mid = (lo + hi) >> 1
            if offsets[mid + 1] <= pos:
                lo = mid + 1
            else:
                hi = mid
        seq_id = lo
        seq_start = int(offsets[seq_id])
        seq_len = int(offsets[seq_id + 1]) - seq_start
        pos_local = pos - seq_start
        H = max(seq_len - int(num_targets[seq_id]), 0)
        if pos_local < H:
            c0 = min(pos_local + 1, eff_k2)
            c1 = max(eff_k2, pos_local - sla_k1 + 1)
            c2 = pos_local + 1
        else:
            c0 = c1 = c2 = H
        af[0, 0, pos, 0] = seq_start
        af[0, 0, pos, 1] = seq_start + c0
        af[0, 0, pos, 2] = seq_start + c1
        af[0, 0, pos, 3] = seq_start + c2
    return af


def make_sequences(seq_lens):
    """jagged 序列长度列表 -> offsets / total_q"""
    offsets = np.zeros(len(seq_lens) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(seq_lens, dtype=np.int32)
    return offsets, int(offsets[-1])


def run_compare(offsets, sla_k1, sla_k2, contextual_seq_len, num_targets, total_seqlen, use_graph=False):
    torch.npu.set_device(DEVICE)
    cpu = build_sla_mask_arbitrary_func_cpu(offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets)
    offsets_t = torch.from_numpy(offsets).to(DEVICE)
    nt_t = torch.from_numpy(num_targets).to(DEVICE) if num_targets is not None else None

    if use_graph:
        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.build_sla_mask_arbitrary_func, dynamic=True, fullgraph=True
        )
        y = compiled_fn(offsets_t, total_seqlen, sla_k1, sla_k2, contextual_seq_len, nt_t)
    else:
        y = ops_rec.ascendc.attention.build_sla_mask_arbitrary_func(
            offsets_t, total_seqlen, sla_k1, sla_k2, contextual_seq_len, nt_t
        )

    y_np = y.cpu().numpy()
    assert y_np.shape == (1, 1, total_seqlen, 4), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), f"mismatch; first diffs: {np.argwhere(y_np != cpu)[:10].tolist()}"


# (name, seq_lens, sla_k1, sla_k2, contextual_seq_len, num_targets or None)
CASES = [
    ("basic", [8], 3, 4, 0, None),
    ("k1_zero", [8], 0, 4, 0, None),
    ("k2_zero_ctx_zero", [8], 3, 0, 0, None),
    ("full_causal", [8], 16, 16, 0, None),
    ("with_target", [6, 4], 2, 3, 0, [2, 1]),
    ("all_target", [8], 3, 4, 0, [8]),
    ("no_target_zero", [8], 3, 4, 0, [0]),
    ("ctx_raise_effk2", [8], 2, 2, 5, None),
    ("multi_batch", [5, 7, 3, 9], 3, 6, 2, [2, 1, 0, 8]),
    ("large_seq", [1000], 32, 128, 0, None),
    ("single_token", [1], 3, 1, 0, None),
    ("seq_len_one_with_tgt", [1], 2, 1, 0, [1]),
]


@pytest.mark.ci
@pytest.mark.parametrize("name,seq_lens,k1,k2,ctx,nt", CASES)
def test_build_sla_mask_arbitrary_func_eager(name, seq_lens, k1, k2, ctx, nt):
    offsets, total = make_sequences(seq_lens)
    run_compare(
        offsets, k1, k2, ctx, np.asarray(nt, dtype=np.int32) if nt is not None else None, total, use_graph=False
    )


@pytest.mark.graph
@pytest.mark.parametrize("name,seq_lens,k1,k2,ctx,nt", CASES)
def test_build_sla_mask_arbitrary_func_graph(name, seq_lens, k1, k2, ctx, nt):
    offsets, total = make_sequences(seq_lens)
    run_compare(offsets, k1, k2, ctx, np.asarray(nt, dtype=np.int32) if nt is not None else None, total, use_graph=True)


@pytest.mark.parametrize("bs", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_sla_mask_arbitrary_func_random(bs, seed):
    """随机多 batch / K1 / K2 / ctx / num_targets 组合，覆盖历史与 target 混排。"""
    rng = np.random.RandomState(bs * 100 + seed)
    seq_lens = rng.randint(1, 12, size=bs).tolist()
    k1 = int(rng.randint(0, 8))
    k2 = int(rng.randint(0, 8))
    ctx = int(rng.randint(0, 5))
    nt = rng.randint(0, 10, size=bs).tolist()
    offsets, total = make_sequences(seq_lens)
    run_compare(offsets, k1, k2, ctx, np.asarray(nt, dtype=np.int32), total, use_graph=False)


def test_build_sla_mask_arbitrary_func_empty():
    offsets = np.array([0], dtype=np.int32)
    cpu = np.zeros((1, 1, 0, 4), dtype=np.int32)
    y = ops_rec.ascendc.attention.build_sla_mask_arbitrary_func(torch.from_numpy(offsets).to(DEVICE), 0, 3, 4, 0, None)
    assert y.shape == (1, 1, 0, 4)
    assert np.array_equal(y.cpu().numpy(), cpu)


def test_build_sla_mask_arbitrary_func_default_nt():
    """不传 num_targets 等价于全 0（全部 history）。"""
    offsets, total = make_sequences([5, 4])
    run_compare(offsets, 2, 3, 0, None, total, use_graph=False)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
