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


def build_hstu_mask_arbitrary_func_mask_cpu(offsets_i32, ctx_lens, tgt_lens, target_group_size, total_seqlen):
    """NumPy reference (v2 layout, n_func=4), output [1, 1, total_seqlen, 4]."""
    arbitrary_func = np.zeros((1, 1, total_seqlen, 4), dtype=np.int32)
    af = arbitrary_func[0, 0, :, :]

    positions = np.arange(total_seqlen, dtype=np.int32)
    seq_id = np.searchsorted(offsets_i32[1:] - 1, positions, side='left')

    seq_start = offsets_i32[seq_id]
    seq_end = offsets_i32[seq_id + 1]
    seq_len = seq_end - seq_start

    ctx_len = ctx_lens[seq_id]
    tgt_len = tgt_lens[seq_id]
    hist_len = seq_len - ctx_len - tgt_len

    pos_in_seq = positions - seq_start
    ctx_end = ctx_len
    hist_end = ctx_len + hist_len
    base = seq_start + hist_end

    is_ctx = pos_in_seq < ctx_end
    is_hist = (pos_in_seq >= ctx_end) & (pos_in_seq < hist_end)
    is_tgt = pos_in_seq >= hist_end

    af[is_ctx, 0] = seq_start[is_ctx]
    af[is_ctx, 1] = base[is_ctx]

    af[is_hist, 0] = seq_start[is_hist]
    af[is_hist, 1] = (positions + 1)[is_hist]

    tgt_idx = pos_in_seq - hist_end
    group_start = (tgt_idx // target_group_size) * target_group_size
    af[is_tgt, 0] = seq_start[is_tgt]
    af[is_tgt, 1] = base[is_tgt]
    af[is_tgt, 2] = (base + group_start)[is_tgt]
    af[is_tgt, 3] = (base + tgt_idx + 1)[is_tgt]

    return arbitrary_func


def make_sequences(batch_size, ctx_lens_list, tgt_lens_list, hist_lens_list):
    assert len(ctx_lens_list) == batch_size
    assert len(tgt_lens_list) == batch_size
    assert len(hist_lens_list) == batch_size
    seq_lens = [c + h + t for c, h, t in zip(ctx_lens_list, hist_lens_list, tgt_lens_list)]
    offsets = np.zeros(batch_size + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(seq_lens, dtype=np.int32)
    total_seqlen = int(offsets[-1])
    return (offsets, np.asarray(ctx_lens_list, dtype=np.int32), np.asarray(tgt_lens_list, dtype=np.int32), total_seqlen)


def run_compare(offsets, ctx, tgt, group_size, total_seqlen, use_graph=False):
    torch.npu.set_device(DEVICE)
    cpu = build_hstu_mask_arbitrary_func_mask_cpu(offsets, ctx, tgt, group_size, total_seqlen)

    if use_graph:
        compiled_fn = torch_compiled(
            ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func, dynamic=True, fullgraph=True
        )
        y = compiled_fn(
            offsets=torch.from_numpy(offsets).to(DEVICE),
            ctx_lens=torch.from_numpy(ctx).to(DEVICE),
            tgt_lens=torch.from_numpy(tgt).to(DEVICE),
            target_group_size=group_size,
            total_seqlen=total_seqlen,
        )
    else:
        y = ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(ctx).to(DEVICE),
            torch.from_numpy(tgt).to(DEVICE),
            group_size,
            total_seqlen,
        )

    y_np = y.cpu().numpy()
    assert y_np.shape == (1, 1, total_seqlen, 4), f"shape {y_np.shape}"
    assert np.array_equal(y_np, cpu), f"mismatch; first diffs: {np.argwhere(y_np != cpu)[:5].tolist()}"


CASES = [
    ("given_bs1", 1, [0], [3072], [1000], 4),
    ("single_full", 1, [5], [6], [4], 3),
    ("no_hist", 1, [3], [4], [0], 2),
    ("all_ctx", 1, [4], [0], [0], 1),
    ("all_hist", 1, [0], [0], [4], 1),
    ("all_tgt", 1, [0], [4], [0], 1),
    ("region_1_1_1", 1, [1], [1], [1], 1),
    ("region_8_8_8", 1, [8], [8], [8], 4),
    ("region_16_1_1", 1, [16], [1], [1], 1),
    ("region_1_16_1", 1, [1], [16], [1], 2),
]


@pytest.mark.ci
@pytest.mark.parametrize("name,bs,cl,tl,hl,g", CASES)
def test_build_hstu_mask_arbitrary_func_eager(name, bs, cl, tl, hl, g):
    offsets, ctx, tgt, total = make_sequences(bs, cl, tl, hl)
    run_compare(offsets, ctx, tgt, g, total, use_graph=False)


@pytest.mark.graph
@pytest.mark.parametrize("name,bs,cl,tl,hl,g", CASES)
def test_build_hstu_mask_arbitrary_func_graph(name, bs, cl, tl, hl, g):
    offsets, ctx, tgt, total = make_sequences(bs, cl, tl, hl)
    run_compare(offsets, ctx, tgt, g, total, use_graph=True)


@pytest.mark.parametrize("bs", [2, 4, 8, 16])
@pytest.mark.parametrize("g", [1, 2, 4])
def test_build_hstu_mask_arbitrary_func_multi_batch(bs, g):
    np.random.seed(bs * 100 + g)
    c = np.random.randint(0, 8, size=bs).tolist()
    t = np.random.randint(0, 8, size=bs).tolist()
    h = np.random.randint(0, 8, size=bs).tolist()
    offsets, ctx, tgt, total = make_sequences(bs, c, t, h)
    run_compare(offsets, ctx, tgt, g, total, use_graph=False)


def test_build_hstu_mask_arbitrary_func_large():
    bs = 100
    c = [500] * bs
    t = [300] * bs
    h = [200] * bs
    offsets, ctx, tgt, total = make_sequences(bs, c, t, h)
    run_compare(offsets, ctx, tgt, 4, total, use_graph=False)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
