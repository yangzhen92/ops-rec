"""Forward/backward local-window precision against the dense CPU golden."""

import os

import pytest
import torch

from utils.data import TestDataGenerator as DataGenerator
from utils.feature_matrix import create_backends
from utils.mask import create_local_window_mask

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("has_rab", (False, True))
@pytest.mark.parametrize("window", ((128, 0), (64, 64), (32, 128), (0, -1), (0, 0), (-1, 64)))
@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_local_window(direction, window, has_rab, dtype, backends):
    batch, heads, max_seq, dim = 2, 2, 256, 64
    offsets = [0, 193, 449]
    generator = DataGenerator(2026093000 + window[0] + max(window[1], 0), True)
    data = generator.gen_data(
        batch,
        heads,
        max_seq,
        max_seq,
        dim,
        dim,
        has_rab,
        dtype,
        window_size=window,
        seq_lens_q_override=(193, 256),
        seq_lens_k_override=(193, 256),
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    dense_mask = torch.zeros(batch, heads, max_seq, max_seq, dtype=dtype)
    for b, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        length = end - begin
        dense_mask[b, :, :length, :length] = create_local_window_mask(length, length, *window).to(dtype)
    fused = backends[0].kernel(0.5, 1 / max_seq, has_rab, max_seq, max_seq, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / max_seq, has_rab, max_seq, max_seq, offsets_q, offsets_k)
    if direction == "forward":
        actual = fused.forward(q, k, v, rab, None, window_size=window).cpu()
        expected = native.forward(q, k, v, rab, dense_mask)
        passed, detail = backends[2].forward_verify(actual, expected)
    else:
        actual = fused.backward(grad, q, k, v, rab, None, window_size=window)
        expected = native.backward(grad, q, k, v, rab, dense_mask)
        passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail


@pytest.mark.parametrize("direction", ("forward", "backward"))
@pytest.mark.parametrize("window", ((64, 0), (32, 128)))
def test_local_window_delta_q(direction, window, backends):
    batch, heads, max_q, max_k, dim = 1, 2, 129, 256, 96
    generator = DataGenerator(2026093100 + window[0] + window[1], True)
    data = generator.gen_data(
        batch,
        heads,
        max_q,
        max_k,
        dim,
        dim,
        True,
        torch.float16,
        window_size=window,
        seq_lens_q_override=(129,),
        seq_lens_k_override=(256,),
        rab_heads=1,
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    dense_mask = create_local_window_mask(max_q, max_k, *window).view(1, 1, max_q, max_k)
    dense_mask = dense_mask.expand(batch, heads, max_q, max_k).to(torch.float16)
    fused = backends[0].kernel(0.5, 1 / max_q, True, max_q, max_k, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / max_q, True, max_q, max_k, offsets_q, offsets_k)
    if direction == "forward":
        actual = fused.forward(q, k, v, rab, None, window_size=window).cpu()
        expected = native.forward(q, k, v, rab, dense_mask)
        passed, detail = backends[2].forward_verify(actual, expected)
    else:
        actual = fused.backward(grad, q, k, v, rab, None, window_size=window)
        expected = native.backward(grad, q, k, v, rab, dense_mask)
        passed, detail = backends[2].backward_verify(actual, expected)
    assert passed, detail


@pytest.mark.parametrize("window", ((-2, 0), (0, -2), (256, 0), (0, 256)))
def test_invalid_local_window_rejected(window, backends):
    batch, heads, max_seq, dim = 1, 1, 256, 64
    generator = DataGenerator(2026100100 + abs(window[0]) + abs(window[1]), True)
    data = generator.gen_data(batch, heads, max_seq, max_seq, dim, dim, False, torch.float16)
    _, q, k, v, _, _, offsets_q, offsets_k = data
    fused = backends[0].kernel(0.5, 1 / max_seq, False, max_seq, max_seq, offsets_q, offsets_k)
    with pytest.raises(RuntimeError, match="window_size"):
        fused.forward(q, k, v, None, None, window_size=window)


@pytest.mark.parametrize(
    "num_context,num_target",
    ((32, 0), (0, 32), (32, 32)),
    ids=("context", "target", "context-target"),
)
@pytest.mark.parametrize("direction", ("forward", "backward"))
def test_local_window_causal_combinations(direction, num_context, num_target, backends):
    batch, heads, max_seq, dim = 2, 2, 160, 64
    seq_lens = (97, 160)
    window = (32, 16)
    target_group_size = 8
    generator = DataGenerator(2026100200 + num_context + num_target, True)
    data = generator.gen_data(
        batch,
        heads,
        max_seq,
        max_seq,
        dim,
        dim,
        True,
        torch.float16,
        window_size=window,
        num_context=num_context,
        num_target=num_target,
        target_group_size=target_group_size,
        seq_lens_q_override=seq_lens,
        seq_lens_k_override=seq_lens,
        rab_heads=1,
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    dense_mask = torch.zeros(batch, heads, max_seq, max_seq, dtype=torch.float16)
    for batch_index, length in enumerate(seq_lens):
        local = create_local_window_mask(
            length,
            length,
            *window,
            num_context=num_context,
            num_target=num_target,
            target_group_size=target_group_size,
        )
        dense_mask[batch_index, :, :length, :length] = local
    context = [num_context] * batch if num_context else None
    target = [num_target] * batch if num_target else None
    fused = backends[0].kernel(0.5, 1 / max_seq, True, max_seq, max_seq, offsets_q, offsets_k)
    native = backends[1].kernel(0.5, 1 / max_seq, True, max_seq, max_seq, offsets_q, offsets_k)
    if direction == "forward":
        actual = fused.forward(
            q,
            k,
            v,
            rab,
            None,
            num_context=context,
            num_target=target,
            target_group_size=target_group_size,
            window_size=window,
        ).cpu()
        expected = native.forward(q, k, v, rab, dense_mask)
        passed, detail = backends[2].forward_verify(actual, expected)
    else:
        actual = fused.backward(
            grad,
            q,
            k,
            v,
            rab,
            None,
            window_size=window,
            num_context=context,
            num_target=target,
            target_group_size=target_group_size,
        )
        expected = native.backward(grad, q, k, v, rab, dense_mask)
        passed, detail = backends[2].backward_verify(actual, expected)
        masked = dense_mask[:, :1] == 0
        assert torch.count_nonzero(actual[3][masked]).item() == 0
    assert passed, detail
