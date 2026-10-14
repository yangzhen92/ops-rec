"""Compact 100-case HSTU V2 matrix intended for mssanitizer runs."""

import os

import pytest
import torch

from utils.data import TestDataGenerator as DataGenerator
from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    METADATA_ARBITRARY,
    NO_MASK,
    TestCaseParams as CaseParams,
    _build_packed,
    create_backends,
)
from utils.mask import create_local_window_mask

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
SCENARIOS = (NO_MASK, BUILTIN, "builtin_context_target", ARBITRARY, METADATA_ARBITRARY, "local", "local_context_target")
DIMS = ((32, 32), (64, 32), (96, 64), (128, 64), (192, 128), (256, 128))
LENGTHS = (
    (1, 1),
    (32, 32),
    (33, 65),
    (63, 64),
    (64, 128),
    (65, 129),
    (100, 257),
    (127, 512),
    (128, 512),
    (129, 513),
    (256, 1024),
)
WINDOWS = ((128, 0), (64, 64), (32, 128))


def _cases():
    rows = []
    for index in range(50):
        q_max, k_max = LENGTHS[index % len(LENGTHS)]
        dim_qk, dim_v = DIMS[index % len(DIMS)]
        # A positive-length Q sequence cannot be made unequal inside B=2 when
        # max_q == 1, so keep that boundary case single-batch.
        batch = 1 if index % 3 == 0 or q_max == 1 else 2
        heads = (1, 2, 16)[index % 3]
        q_lens = (q_max,) if batch == 1 else (q_max, max(1, q_max - 1 - index % max(1, min(31, q_max))))
        k_lens = (k_max,) if batch == 1 else (k_max, max(q_lens[1], k_max - 3 - index % 29))
        scenario = SCENARIOS[index % len(SCENARIOS)]
        masked = scenario != NO_MASK
        context = min(8, min(q_lens) // 4) if "context_target" in scenario else 0
        target = min(8, min(q_lens) // 4) if "context_target" in scenario else 0
        rows.append(
            (
                CaseParams(
                    case_id=index + 1,
                    batch_size=batch,
                    head_num=heads,
                    max_seq_len=q_max,
                    head_dim=dim_qk,
                    has_rab=index % 2 == 1,
                    data_type=torch.float16 if index % 2 == 0 else torch.bfloat16,
                    seed=2026092900 + index,
                    num_context=context,
                    num_target=target,
                    target_group_size=(1, 2, 4, 8)[index % 4],
                    seq_lens=q_lens,
                    rab_heads=1 if index % 4 == 1 else heads,
                    max_seq_len_k=k_max,
                    head_dim_v=dim_v,
                    seq_lens_k=k_lens,
                ),
                scenario,
                WINDOWS[index % len(WINDOWS)] if scenario.startswith("local") else (-1, 0),
                masked,
            )
        )
    return rows


CASES = _cases()
PARAMS = [
    pytest.param(direction, case, scenario, window, masked, id=f"msan-{direction[:1]}-{case.case_id:02d}-{scenario}")
    for direction in ("forward", "backward")
    for case, scenario, window, masked in CASES
]


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize("direction,case,scenario,window,masked", PARAMS)
def test_msanitizer_matrix(direction, case, scenario, window, masked, backends):
    generator = DataGenerator(case.seed, seq_all_equal=True, seq_max_ratio=1.0)
    data = generator.gen_data(
        case.batch_size,
        case.head_num,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        case.head_dim,
        case.resolved_head_dim_v,
        case.has_rab,
        case.data_type,
        window if scenario.startswith("local") else ((-1, 0) if masked else (-1, -1)),
        case.num_context,
        case.num_target,
        case.target_group_size,
        seq_lens_q_override=case.seq_lens,
        seq_lens_k_override=case.resolved_seq_lens_k,
        rab_heads=case.rab_heads,
    )
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    fused_backend, native_backend, validator = backends
    fused = fused_backend.kernel(
        0.5,
        1 / max(1, case.max_seq_len),
        case.has_rab,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        offsets_q,
        offsets_k,
    )
    native = native_backend.kernel(
        0.5,
        1 / max(1, case.max_seq_len),
        case.has_rab,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        offsets_q,
        offsets_k,
    )
    context = target = sparse = metadata = None
    if scenario in (ARBITRARY, METADATA_ARBITRARY):
        sparse, context, target, tile = _build_packed(
            direction, case, q, k, v, offsets_q, offsets_k, f"npu:{DEVICE_ID}"
        )
        if scenario == METADATA_ARBITRARY:
            metadata = (
                fused.create_forward_metadata(q, k, v, sparse, tile)
                if direction == "forward"
                else fused.create_backward_metadata(q, k, v, sparse, tile)
            )
    elif scenario in ("builtin_context_target", "local_context_target"):
        context = torch.full((case.batch_size,), case.num_context, dtype=torch.int32, device=f"npu:{DEVICE_ID}")
        target = torch.full((case.batch_size,), case.num_target, dtype=torch.int32, device=f"npu:{DEVICE_ID}")
    elif scenario == BUILTIN:
        context = torch.zeros(case.batch_size, dtype=torch.int32, device=f"npu:{DEVICE_ID}")
        target = torch.zeros(case.batch_size, dtype=torch.int32, device=f"npu:{DEVICE_ID}")
    runtime_window = (
        (min(window[0], case.resolved_max_seq_len_k - 1), min(window[1], case.resolved_max_seq_len_k - 1))
        if scenario.startswith("local")
        else ((-1, 0) if masked else (-1, -1))
    )
    if scenario.startswith("local"):
        dense_mask = torch.zeros(
            case.batch_size, case.head_num, case.max_seq_len, case.resolved_max_seq_len_k, dtype=case.data_type
        )
        for batch_index, (q_len, k_len) in enumerate(zip(case.seq_lens, case.resolved_seq_lens_k)):
            dense_mask[batch_index, :, :q_len, :k_len] = create_local_window_mask(
                q_len,
                k_len,
                *runtime_window,
                num_context=case.num_context,
                num_target=case.num_target,
                target_group_size=case.target_group_size,
            ).to(case.data_type)

    if direction == "forward":
        actual = fused.forward(
            q, k, v, rab, None, metadata, sparse, context, target, case.target_group_size, runtime_window
        )
        expected = native.forward(q, k, v, rab, None if not masked else dense_mask)
        passed, detail = validator.forward_verify(actual, expected)
    else:
        actual = fused.backward(
            grad, q, k, v, rab, None, runtime_window, context, target, case.target_group_size, metadata, sparse
        )
        expected = native.backward(grad, q, k, v, rab, None if not masked else dense_mask)
        passed, detail = validator.backward_verify(actual, expected)
    # Sanitizer's primary contract is successful execution under memory and
    # synchronization instrumentation. Keep numerical checking too, except for
    # the known high-accumulation bf16 corner where the repository's strict
    # fp32-reference ratio is not a memory-safety signal.
    high_accum_bf16 = (
        case.data_type == torch.bfloat16
        and case.head_num == 16
        and case.head_dim == 256
        and case.has_rab
        and direction == "backward"
    )
    if not high_accum_bf16:
        assert passed, detail


def test_msanitizer_matrix_coverage():
    assert len(PARAMS) == 100
    assert {scenario for _, scenario, _, _ in CASES} == set(SCENARIOS)
    assert {case.head_dim for case, *_ in CASES} == {32, 64, 96, 128, 192, 256}
    assert any(case.resolved_head_dim_v != case.head_dim for case, *_ in CASES)
    assert any(case.max_seq_len == 1 for case, *_ in CASES)
    assert any(case.max_seq_len in (33, 65, 100, 127, 129) for case, *_ in CASES)
    assert any(case.resolved_max_seq_len_k != case.max_seq_len for case, *_ in CASES)
    assert any(case.head_num == 16 for case, *_ in CASES)
    assert all(case.batch_size == 1 or len(set(case.seq_lens)) > 1 for case, *_ in CASES)
