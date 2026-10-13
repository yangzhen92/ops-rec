"""Guide9 precision boundaries not covered by the main feature matrix."""

import os

import pytest
import torch

from utils.data import TestDataGenerator as DataGenerator
from utils.feature_matrix import NO_MASK, TestCaseParams, create_backends, run_precision_case
from utils.mask import create_local_window_mask

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
TestCaseParams.__test__ = False


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


def _build_bwd_plan(q, k, v, af, offsets_q, offsets_k, max_q, max_k):
    context = torch.zeros(offsets_q.numel() - 1, dtype=torch.int32, device=offsets_q.device)
    target = torch.zeros_like(context)
    tile, sparse, _, _ = torch.ops.ops_rec.get_bwd_attn_plan(
        q.to(offsets_q.device),
        k.to(offsets_q.device),
        v.to(offsets_q.device),
        af,
        offsets_q,
        offsets_k,
        max_q,
        max_k,
        context,
        target,
        1,
        -1,
        0,
        0,
        True,
    )
    return list(tile), list(sparse), context, target


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("alpha,scale", ((1.0e-4, 1.0e-8), (0.5, 1.0), (2.0, 16.0)))
def test_scale_alpha_extremes(direction, dtype, alpha, scale, backends):
    case = TestCaseParams(901, 2, 1, 65, 64, False, dtype, 2026093001, 0, 0, 1, (65, 33), None, 129, 32, (129, 100))
    passed, detail = run_precision_case(
        direction, case, NO_MASK, *backends, f"npu:{DEVICE_ID}", alpha=alpha, scale=scale
    )
    assert passed, detail


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rab_value", (-20.0, 0.0, 20.0))
def test_rab_saturation_values(dtype, rab_value, backends):
    q_len, k_len, dim = 33, 65, 32
    data = DataGenerator(2026093002 + int(rab_value), True).gen_data(
        1,
        1,
        q_len,
        k_len,
        dim,
        dim,
        True,
        dtype,
        window_size=(-1, 0),
        seq_lens_q_override=(q_len,),
        seq_lens_k_override=(k_len,),
        rab_heads=1,
    )
    _, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    rab.fill_(rab_value)
    fused = backends[0].kernel(0.5, 1 / q_len, True, q_len, k_len, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / q_len, True, q_len, k_len, offsets_q, offsets_k)
    actual = fused.forward(q, k, v, rab, None, num_context=[0], num_target=[0], target_group_size=1)
    expected = native.forward(q, k, v, rab, dense_mask)
    passed, detail = backends[2].forward_verify(actual, expected)
    assert passed, detail


def test_local_masked_columns_have_exact_zero_dk_dv_and_drab(backends):
    q_len, k_len, dim = 65, 257, 64
    window = (0, 0)
    data = DataGenerator(2026093003, True).gen_data(
        1,
        2,
        q_len,
        k_len,
        dim,
        dim,
        True,
        torch.float16,
        window_size=window,
        seq_lens_q_override=(q_len,),
        seq_lens_k_override=(k_len,),
        rab_heads=1,
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    fused = backends[0].kernel(0.5, 1 / q_len, True, q_len, k_len, offsets_q, offsets_k)
    dq, dk, dv, drab = fused.backward(grad, q, k, v, rab, None, window_size=window)
    mask = create_local_window_mask(q_len, k_len, *window).bool()
    fully_masked_k = ~mask.any(dim=0)
    assert fully_masked_k.any()
    assert torch.count_nonzero(dk[fully_masked_k]).item() == 0
    assert torch.count_nonzero(dv[fully_masked_k]).item() == 0
    assert torch.count_nonzero(drab[0, 0][~mask]).item() == 0
    # Every Q row has one visible K position, so dQ is intentionally not
    # asserted zero merely because individual score positions are masked.
    assert torch.isfinite(dq).all()


def test_mixed_rab_float32_is_rejected(backends):
    q_len, dim = 33, 32
    _, q, k, v, _, _, offsets_q, offsets_k = DataGenerator(2026093004, True).gen_data(
        1, 1, q_len, q_len, dim, dim, False, torch.float16, seq_lens_q_override=(q_len,), seq_lens_k_override=(q_len,)
    )
    rab = torch.zeros(1, 1, q_len, q_len, dtype=torch.float32)
    fused = backends[0].kernel(0.5, 1 / q_len, True, q_len, q_len, offsets_q, offsets_k)
    with pytest.raises(RuntimeError, match="rab must be float16 or bfloat16"):
        fused.forward(q, k, v, rab, None)


def test_fully_masked_arbitrary_gradients_are_exact_zero(backends):
    length, heads, dim = 64, 2, 32
    data = DataGenerator(2026093005, True).gen_data(
        1,
        heads,
        length,
        length,
        dim,
        dim,
        True,
        torch.float16,
        window_size=(-1, 0),
        seq_lens_q_override=(length,),
        seq_lens_k_override=(length,),
        rab_heads=1,
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    offsets = torch.tensor(offsets_q, dtype=torch.int32, device="npu")
    af = torch.zeros(1, 1, length, 4, dtype=torch.int32, device="npu")
    _, sparse, context, target = _build_bwd_plan(q, k, v, af, offsets, offsets, length, length)
    fused = backends[0].kernel(0.5, 1 / length, True, length, length, offsets_q, offsets_k)
    result = fused.backward(grad, q, k, v, rab, None, (-1, 0), context, target, 1, None, sparse)
    for name, tensor in zip(("dq", "dk", "dv", "drab"), result):
        assert tensor is not None
        assert torch.count_nonzero(tensor).item() == 0, (name, torch.count_nonzero(tensor).item(), tensor.abs().max())


@pytest.mark.parametrize("use_metadata", (False, True))
def test_partially_empty_arbitrary_k_tile_is_exact_zero(use_metadata, backends):
    q_len, k_len, heads, dim = 64, 256, 2, 32
    data = DataGenerator(2026093006, True).gen_data(
        1,
        heads,
        q_len,
        k_len,
        dim,
        dim,
        False,
        torch.float16,
        seq_lens_q_override=(q_len,),
        seq_lens_k_override=(k_len,),
    )
    grad, q, k, v, _, _, offsets_q, offsets_k = data
    offsets_q_npu = torch.tensor(offsets_q, dtype=torch.int32, device="npu")
    offsets_k_npu = torch.tensor(offsets_k, dtype=torch.int32, device="npu")
    # Every Q row attends only K[0:128]; the second physical K tile is empty.
    af = torch.zeros(k_len, 2, dtype=torch.int32, device="npu")
    af[:, 1] = 128
    tile, sparse, context, target = _build_bwd_plan(q, k, v, af, offsets_q_npu, offsets_k_npu, q_len, k_len)
    fused = backends[0].kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
    metadata = fused.create_backward_metadata(q, k, v, sparse, tile) if use_metadata else None
    actual = fused.backward(grad, q, k, v, None, None, (-1, 0), context, target, 1, metadata, sparse)
    dense_mask = torch.zeros(1, heads, q_len, k_len, dtype=torch.float16)
    dense_mask[:, :, :, :128] = 1
    expected = (
        backends[1]
        .kernel(0.5, 1 / q_len, False, q_len, k_len, offsets_q, offsets_k)
        .backward(grad, q, k, v, None, dense_mask)
    )
    passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail
    assert torch.count_nonzero(actual[1][128:]).item() == 0
    assert torch.count_nonzero(actual[2][128:]).item() == 0
    assert torch.count_nonzero(actual[1][:128]).item() > 0
    assert torch.count_nonzero(actual[2][:128]).item() > 0


@pytest.mark.parametrize("use_metadata", (False, True))
def test_multibatch_ragged_empty_k_tiles_are_exact_zero(use_metadata, backends):
    q_lens, k_lens, heads = (65, 33), (257, 193), 2
    dim_qk, dim_v = 64, 32
    data = DataGenerator(2026093007, True).gen_data(
        2,
        heads,
        max(q_lens),
        max(k_lens),
        dim_qk,
        dim_v,
        False,
        torch.bfloat16,
        seq_lens_q_override=q_lens,
        seq_lens_k_override=k_lens,
    )
    grad, q, k, v, _, _, offsets_q, offsets_k = data
    offsets_q_npu = torch.tensor(offsets_q, dtype=torch.int32, device="npu")
    offsets_k_npu = torch.tensor(offsets_k, dtype=torch.int32, device="npu")
    af = torch.zeros(sum(k_lens), 2, dtype=torch.int32, device="npu")
    # Batch 0 exposes K[0:128], batch 1 exposes K[0:64]. Remaining K tiles
    # (including the ragged tail) must be initialized by the device zero path.
    af[: k_lens[0], 1] = 128
    af[k_lens[0] :, 0] = k_lens[0]
    af[k_lens[0] :, 1] = k_lens[0] + 64
    tile, sparse, context, target = _build_bwd_plan(q, k, v, af, offsets_q_npu, offsets_k_npu, max(q_lens), max(k_lens))
    fused = backends[0].kernel(0.5, 1 / max(q_lens), False, max(q_lens), max(k_lens), offsets_q, offsets_k)
    metadata = fused.create_backward_metadata(q, k, v, sparse, tile) if use_metadata else None
    actual = fused.backward(grad, q, k, v, None, None, (-1, 0), context, target, 1, metadata, sparse)
    dense_mask = torch.zeros(2, heads, max(q_lens), max(k_lens), dtype=torch.bfloat16)
    dense_mask[0, :, : q_lens[0], :128] = 1
    dense_mask[1, :, : q_lens[1], :64] = 1
    expected = (
        backends[1]
        .kernel(0.5, 1 / max(q_lens), False, max(q_lens), max(k_lens), offsets_q, offsets_k)
        .backward(grad, q, k, v, None, dense_mask)
    )
    passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail
    for tensor in actual[1:3]:
        assert torch.count_nonzero(tensor[128:257]).item() == 0
        assert torch.count_nonzero(tensor[257 + 64 :]).item() == 0
