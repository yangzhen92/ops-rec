"""Regression coverage for guide7 correctness fixes."""

import os

import pytest
import numpy as np
import torch

from utils.data import TestDataGenerator as DataGenerator
from utils.feature_matrix import BUILTIN, TestCaseParams, create_backends, run_precision_case
from utils.mask import create_causal_mask

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
TestCaseParams.__test__ = False


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_causal_short_k_long_q(direction, backends):
    q_len, k_len, heads, dim = 257, 129, 2, 64
    torch.manual_seed(2026100701)
    grad = torch.empty((q_len, heads, dim), dtype=torch.float16).uniform_(-1, 1)
    q = torch.empty((q_len, heads, dim), dtype=torch.float16).uniform_(-1, 1)
    k = torch.empty((k_len, heads, dim), dtype=torch.float16).uniform_(-1, 1)
    v = torch.empty((k_len, heads, dim), dtype=torch.float16).uniform_(-1, 1)
    offsets_q = np.asarray([0, q_len], dtype=np.int32)
    offsets_k = np.asarray([0, k_len], dtype=np.int32)
    dense_mask = (
        create_causal_mask(q_len, k_len).to(torch.float16).view(1, 1, q_len, k_len).expand(1, heads, q_len, k_len)
    )
    fused = backends[0].kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
    if direction == "forward":
        actual = fused.forward(q, k, v, None, None, num_context=[0], window_size=(-1, 0)).cpu()
        expected = native.forward(q, k, v, None, dense_mask)
        passed, detail = backends[2].forward_verify(actual, expected)
    else:
        actual = fused.backward(grad, q, k, v, None, None, window_size=(-1, 0))
        expected = native.backward(grad, q, k, v, None, dense_mask)
        passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail


def test_negative_scale_backward(backends):
    case = TestCaseParams(701, 1, 2, 256, 64, True, torch.float16, 2026100702, 0, 0, 1)
    passed, detail = run_precision_case(
        "backward",
        case,
        BUILTIN,
        *backends,
        f"npu:{DEVICE_ID}",
        alpha=0.5,
        scale=-1 / 256,
    )
    assert passed, detail


def test_qshare_all_heads_written(backends):
    case = TestCaseParams(702, 1, 4, 128, 64, False, torch.float16, 2026100703, 0, 0, 1)
    generator = DataGenerator(case.seed, True)
    data = generator.gen_data(1, 4, 128, 128, 64, 64, False, torch.float16)
    grad, q, k, v, _, _, offsets_q, offsets_k = data
    fused = backends[0].kernel(0.5, 1 / 128, False, 128, 128, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / 128, False, 128, 128, offsets_q, offsets_k)
    actual = fused.backward(grad, q, k, v, None, None, window_size=(-1, -1))
    expected = native.backward(grad, q, k, v, None, None)
    passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail
    assert all(torch.count_nonzero(actual[0][:, head]).item() > 0 for head in range(case.head_num))
