"""Guide-new8 coverage for non-physical runtime attention tiles."""

import os

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    NO_MASK,
    TestCaseParams,
    create_backends,
    run_precision_case,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("pipeline", (NO_MASK, BUILTIN, ARBITRARY))
@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            TestCaseParams(
                801, 1, 1, 256, 32, False, torch.float16, 2026100801, 0, 0, 1, (256,), None, 1024, 32, (1024,)
            ),
            id="fp16-d32-n512",
        ),
        pytest.param(
            TestCaseParams(802, 1, 1, 256, 64, True, torch.bfloat16, 2026100802, 0, 0, 1, (256,), 1, 1024, 64, (1024,)),
            id="bf16-d64-n256-rab1",
        ),
    ),
)
def test_forward_large_runtime_n(case, pipeline, backends):
    fused, native, validator = backends
    q = torch.empty((case.max_seq_len, case.head_num, case.head_dim), dtype=case.data_type)
    k = torch.empty((case.resolved_max_seq_len_k, case.head_num, case.head_dim), dtype=case.data_type)
    expected_n = 512 if case.head_dim == 32 else 256
    assert torch.ops.ops_rec.get_fwd_tile_size(q, k, k, True) == [32, expected_n, case.head_dim]
    passed, detail = run_precision_case("forward", case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail


@pytest.mark.parametrize("pipeline", (NO_MASK, BUILTIN, ARBITRARY))
@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            TestCaseParams(
                803, 1, 1, 1024, 32, False, torch.float16, 2026100803, 0, 0, 1, (1024,), None, 1024, 32, (1024,)
            ),
            id="fp16-d32-m512",
        ),
        pytest.param(
            TestCaseParams(
                804, 1, 1, 1024, 64, True, torch.bfloat16, 2026100804, 0, 0, 1, (1024,), 1, 1024, 64, (1024,)
            ),
            id="bf16-d64-m512-rab1",
        ),
    ),
)
def test_backward_large_runtime_m(case, pipeline, backends):
    fused, native, validator = backends
    q = torch.empty((case.max_seq_len, case.head_num, case.head_dim), dtype=case.data_type)
    k = torch.empty((case.resolved_max_seq_len_k, case.head_num, case.head_dim), dtype=case.data_type)
    tile = torch.ops.ops_rec.get_bwd_tile_size(q, k, k, True)
    assert tile[0] == 512 and tile[1] in (64, 128)
    passed, detail = run_precision_case("backward", case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail
