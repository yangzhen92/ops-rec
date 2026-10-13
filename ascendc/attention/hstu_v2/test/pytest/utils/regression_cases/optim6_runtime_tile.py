"""Guide6 runtime-Q-tile precision coverage for long-K/short-Q attention."""

import os

import pytest
import torch

from utils.data import TestDataGenerator as DataGenerator
from utils.feature_matrix import create_backends

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize(
    "q_len,k_len,heads,dim,dtype",
    (
        (128, 512, 1, 32, torch.float16),
        (256, 1024, 1, 64, torch.bfloat16),
        (256, 1024, 1, 128, torch.float16),
        (512, 2048, 1, 128, torch.bfloat16),
        (256, 1024, 4, 128, torch.float16),
        (128, 512, 1, 256, torch.float16),
        (512, 2048, 1, 256, torch.bfloat16),
    ),
)
def test_long_k_short_q_runtime_tile(q_len, k_len, heads, dim, dtype, backends):
    generator = DataGenerator(2026100600 + q_len + dim + heads, True)
    data = generator.gen_data(
        1,
        heads,
        q_len,
        k_len,
        dim,
        dim,
        False,
        dtype,
        seq_lens_q_override=(q_len,),
        seq_lens_k_override=(k_len,),
    )
    _, q, k, v, _, _, offsets_q, offsets_k = data
    fused = backends[0].kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
    actual = fused.forward(q, k, v, None, None).cpu()
    expected = native.forward(q, k, v, None, None)
    passed, detail = backends[2].forward_verify(actual, expected)
    assert passed, detail


def test_runtime_tile_selector_directions_and_short_k():
    torch.npu.set_device(DEVICE_ID)
    q = torch.empty((256, 1, 128), dtype=torch.float16)
    k = torch.empty((1024, 1, 128), dtype=torch.float16)
    assert torch.ops.ops_rec.get_fwd_tile_size(q, k, q, True) == [32, 128, 128]
    assert torch.ops.ops_rec.get_bwd_tile_size(q, k, q, True) == [256, 64, 128]

    short_q = torch.empty((512, 1, 128), dtype=torch.float16)
    short_k = torch.empty((512, 1, 128), dtype=torch.float16)
    assert torch.ops.ops_rec.get_fwd_tile_size(short_q, short_k, short_k, True) == [128, 128, 128]

    long_k = torch.empty((4096, 1, 128), dtype=torch.float16)
    assert torch.ops.ops_rec.get_bwd_tile_size(q, long_k, long_k, True) == [256, 128, 128]
    medium_k = torch.empty((2048, 1, 128), dtype=torch.float16)
    assert torch.ops.ops_rec.get_bwd_tile_size(q, medium_k, medium_k, True) == [256, 64, 128]
    short_bwd_k = torch.empty((512, 1, 128), dtype=torch.float16)
    assert torch.ops.ops_rec.get_bwd_tile_size(q, short_bwd_k, short_bwd_k, True) == [256, 64, 128]

    narrow = torch.empty((1024, 1, 32), dtype=torch.float16)
    assert torch.ops.ops_rec.get_fwd_tile_size(q[:, :, :32], narrow, narrow, True) == [32, 512, 32]
    assert torch.ops.ops_rec.get_bwd_tile_size(q[:, :, :32], narrow, narrow, True) == [256, 64, 32]
    long_q = torch.empty((1024, 1, 32), dtype=torch.float16)
    long_k_narrow = torch.empty((1024, 1, 32), dtype=torch.float16)
    assert torch.ops.ops_rec.get_bwd_tile_size(long_q, long_k_narrow, long_k_narrow, True) == [512, 64, 32]


def test_static_tile_query_does_not_require_an_npu_context():
    q = torch.empty((128, 1, 32), dtype=torch.float16)
    assert torch.ops.ops_rec.get_fwd_tile_size(q, q, q, False) == [128, 128, 128]
    assert torch.ops.ops_rec.get_bwd_tile_size(q, q, q, False) == [256, 128, 128]
