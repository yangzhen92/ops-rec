"""Bounded, deterministic HSTU V2 matrices shared by the four pytest files."""

from dataclasses import dataclass, replace
from itertools import product
import random
import sysconfig

import pytest
import torch
import torch_npu  # noqa: F401

import ops_rec  # noqa: F401
from backend import KernelBackend, create_hstu_atten_backend
from utils.data import TestDataGenerator
from utils.mask import create_causal_mask


METADATA_ARBITRARY = "metadata_arbitrary"
ARBITRARY = "arbitrary"
BUILTIN = "builtin"
NO_MASK = "no_mask"
PLAN = "plan"
PIPELINES = (METADATA_ARBITRARY, ARBITRARY, BUILTIN, NO_MASK, PLAN)
BATCHES = (1, 8, 16)
HEADS = (1, 4, 6)
SEQUENCES = (1, 32, 64, 128, 192, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192)
HEAD_DIMS = (32, 64, 128, 256)
RAB_OPTIONS = (False, True)
DTYPES = (torch.float16, torch.bfloat16)
MATRIX_SEED = 20260922
PRECISION_CONFIGS = 30
BENCHMARK_CONFIGS = 12


@dataclass(frozen=True)
class TestCaseParams:
    """One HSTU self-attention configuration."""

    case_id: int
    batch_size: int
    head_num: int
    max_seq_len: int
    head_dim: int
    has_rab: bool
    data_type: torch.dtype
    seed: int
    num_context: int
    num_target: int
    target_group_size: int
    seq_lens: tuple[int, ...] | None = None
    rab_heads: int | None = None
    max_seq_len_k: int | None = None
    head_dim_v: int | None = None
    seq_lens_k: tuple[int, ...] | None = None

    @property
    def dtype_name(self):
        return "fp16" if self.data_type == torch.float16 else "bf16"

    @property
    def resolved_max_seq_len_k(self):
        return self.max_seq_len if self.max_seq_len_k is None else self.max_seq_len_k

    @property
    def resolved_head_dim_v(self):
        return self.head_dim if self.head_dim_v is None else self.head_dim_v

    @property
    def resolved_seq_lens_k(self):
        return self.seq_lens if self.seq_lens_k is None else self.seq_lens_k


def _make_case(case_id, batch, heads, seq, dim, has_rab, dtype=None, rab_heads=None):
    return TestCaseParams(
        case_id=case_id,
        batch_size=batch,
        head_num=heads,
        max_seq_len=seq,
        head_dim=dim,
        has_rab=has_rab,
        data_type=dtype or DTYPES[(case_id - 1) % len(DTYPES)],
        seed=MATRIX_SEED + case_id,
        num_context=0,
        num_target=0,
        target_group_size=(1, 2, 4, 8)[(case_id - 1) % 4],
        rab_heads=rab_heads,
    )


def _candidate_pool():
    """Keep dense precision references bounded while covering every requested axis."""
    pool = []
    case_id = 0
    sequences = [value for value in SEQUENCES if value <= 2048]
    axes = product(sequences, BATCHES, HEADS, HEAD_DIMS, RAB_OPTIONS, DTYPES)
    for seq, batch, heads, dim, has_rab, dtype in axes:
        case_id += 1
        pool.append(
            _make_case(
                case_id,
                batch,
                heads,
                seq,
                dim,
                has_rab,
                dtype,
                rab_heads=1 if has_rab and heads > 1 else None,
            )
        )
    return pool


def _values(case):
    return (
        case.batch_size,
        case.head_num,
        case.max_seq_len,
        case.head_dim,
        case.has_rab,
        case.data_type,
        case.rab_heads,
    )


def _pairwise_score(candidate, selected):
    values = _values(candidate)
    seen = set()
    for old in selected:
        old_values = _values(old)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                seen.add((left, old_values[left], right, old_values[right]))
    return sum(
        (left, values[left], right, values[right]) not in seen
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _select_cases(count):
    required = [
        _make_case(1, 1, 1, 8192, 32, False, torch.float16),
        _make_case(2, 1, 1, 8192, 64, True, torch.bfloat16),
        _make_case(3, 1, 4, 4096, 256, False, torch.float16),
        _make_case(4, 1, 1, 1, 32, False, torch.float16),
        _make_case(5, 1, 4, 32, 64, True, torch.bfloat16, rab_heads=1),
        _make_case(6, 8, 1, 64, 128, False, torch.bfloat16),
        _make_case(7, 16, 4, 128, 256, True, torch.float16),
        _make_case(8, 8, 4, 192, 64, False, torch.float16),
        _make_case(9, 16, 6, 256, 32, True, torch.bfloat16, rab_heads=1),
        _make_case(10, 1, 6, 512, 128, True, torch.float16, rab_heads=1),
        _make_case(11, 8, 6, 1024, 256, False, torch.bfloat16),
        _make_case(12, 16, 4, 2048, 32, False, torch.float16),
        _make_case(13, 1, 1, 3072, 64, False, torch.bfloat16),
        _make_case(14, 1, 1, 6144, 128, False, torch.float16),
        _make_case(15, 1, 4, 4096, 64, True, torch.bfloat16, rab_heads=1),
    ]
    rng = random.Random(MATRIX_SEED + count)
    pool = [item for item in _candidate_pool() if _values(item) not in {_values(case) for case in required}]
    rng.shuffle(pool)
    selected = list(required)
    while len(selected) < count:
        sample = pool if len(pool) <= 256 else rng.sample(pool, 256)
        best = max(sample, key=lambda item: _pairwise_score(item, selected))
        selected.append(replace(best, case_id=len(selected) + 1, seed=MATRIX_SEED + len(selected) + 1))
        pool.remove(best)
    # Guide8 rebalances 60% of the main matrix toward delta-QK and unequal
    # QK/V dimensions without increasing the case count. Multi-batch variants
    # also use deterministic, non-uniform per-batch lengths.
    rebalanced = []
    remaining = min(18, count)
    dim_v_map = {32: 64, 64: 32, 128: 64, 256: 128}
    for case in selected:
        if remaining and 1 < case.max_seq_len < 8192:
            q_lens = (
                tuple(
                    case.max_seq_len - (index * (case.max_seq_len - 1) // max(1, case.batch_size - 1))
                    for index in range(case.batch_size)
                )
                if case.batch_size > 1
                else (case.max_seq_len,)
            )
            k_lens = tuple(length + case.max_seq_len for length in q_lens)
            case = replace(
                case,
                max_seq_len_k=case.max_seq_len * 2,
                head_dim_v=dim_v_map[case.head_dim],
                seq_lens=q_lens,
                seq_lens_k=k_lens,
            )
            remaining -= 1
        rebalanced.append(case)
    return rebalanced


def precision_cases():
    return _select_cases(PRECISION_CONFIGS)


def ragged_precision_cases():
    """Adversarial batches whose sequence lengths span tile/C0 boundaries."""
    specs = (
        (8, 4, 1024, 64, False, torch.float16, (32, 64, 96, 128, 256, 512, 768, 1024)),
        (8, 1, 1024, 128, True, torch.bfloat16, (1024, 32, 512, 64, 256, 96, 992, 128)),
        (16, 1, 512, 128, False, torch.bfloat16, (32, 512) * 8),
        (
            16,
            4,
            192,
            256,
            True,
            torch.float16,
            (32, 64, 96, 128, 160, 192, 32, 192, 64, 160, 96, 128, 32, 192, 64, 160),
        ),
        (8, 6, 256, 32, False, torch.bfloat16, (256, 32, 224, 64, 192, 96, 160, 128)),
        (8, 4, 512, 64, True, torch.float16, (32, 480, 64, 256, 128, 96, 512, 32)),
        (
            16,
            6,
            256,
            32,
            True,
            torch.bfloat16,
            (32, 256, 32, 256, 64, 224, 64, 224, 96, 192, 96, 192, 128, 160, 128, 160),
        ),
        (8, 1, 1024, 256, False, torch.float16, (1024, 512, 256, 128, 64, 32, 96, 160)),
        (8, 6, 160, 64, True, torch.bfloat16, (32, 64, 96, 128, 160, 32, 64, 160)),
        (
            16,
            4,
            288,
            128,
            False,
            torch.float16,
            (288, 32, 256, 64, 224, 96, 192, 128, 160, 32, 128, 64, 96, 256, 224, 288),
        ),
        (8, 1, 544, 256, True, torch.float16, (32, 512, 64, 480, 96, 256, 128, 544)),
        (16, 6, 128, 32, False, torch.bfloat16, (32, 64, 96, 128, 32, 64, 96, 128, 32, 64, 96, 128, 32, 64, 96, 128)),
        (8, 4, 1024, 128, True, torch.float16, (32, 1024, 64, 992, 96, 768, 128, 512)),
        (
            16,
            1,
            512,
            64,
            False,
            torch.bfloat16,
            (512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192, 160, 128, 96, 64, 32),
        ),
        (2, 4, 2048, 64, False, torch.float16, (32, 2048)),
        (2, 4, 4096, 128, True, torch.bfloat16, (64, 4096)),
        (2, 1, 8192, 256, False, torch.float16, (32, 8192)),
    )
    return [
        replace(
            _make_case(
                PRECISION_CONFIGS + index,
                batch,
                heads,
                seq,
                dim,
                has_rab,
                dtype,
                rab_heads=1 if has_rab and heads > 1 and index % 2 == 0 else None,
            ),
            seq_lens=seq_lens,
        )
        for index, (batch, heads, seq, dim, has_rab, dtype, seq_lens) in enumerate(specs, 1)
    ]


def benchmark_cases():
    return _select_cases(BENCHMARK_CONFIGS)


def case_id(case, prefix):
    return (
        f"{prefix}{case.case_id:03d}-b{case.batch_size}-h{case.head_num}-"
        f"s{case.max_seq_len}-d{case.head_dim}-{case.dtype_name}-rab{int(case.has_rab)}"
    )


def params(cases, prefix):
    selected = cases()
    return pytest_params(selected, prefix)


def pytest_params(cases, prefix):
    return [pytest.param(case, id=case_id(case, prefix)) for case in cases]


def create_backends(device_id):
    fused = create_hstu_atten_backend(
        KernelBackend.ASCEND_FUSE,
        device=device_id,
        ops_library_dir=f"{sysconfig.get_path('purelib')}/ops_rec/ops_rec_atlas_950.so",
    )
    native = create_hstu_atten_backend(KernelBackend.PYTORCH_NATIVE)
    return fused, native, native.validator()


def _make_case_data(case):
    generator = TestDataGenerator(case.seed, seq_all_equal=True, seq_max_ratio=1.0)
    data = generator.gen_data(
        case.batch_size,
        case.head_num,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        case.head_dim,
        case.resolved_head_dim_v,
        case.has_rab,
        case.data_type,
        (-1, 0),
        case.num_context,
        case.num_target,
        case.target_group_size,
        seq_lens_q_override=case.seq_lens,
        seq_lens_k_override=case.resolved_seq_lens_k,
        rab_heads=case.rab_heads,
    )
    grad, q, k, v, rab, _, offsets_q, offsets_k = data
    dense_mask = torch.zeros(
        case.batch_size,
        case.head_num,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        dtype=case.data_type,
    )
    for batch_index, (q_begin, q_end, k_begin, k_end) in enumerate(
        zip(offsets_q[:-1], offsets_q[1:], offsets_k[:-1], offsets_k[1:])
    ):
        q_len = int(q_end - q_begin)
        k_len = int(k_end - k_begin)
        dense_mask[batch_index, :, :q_len, :k_len] = create_causal_mask(
            q_len,
            k_len,
            case.num_context,
            case.num_target,
            case.target_group_size,
        ).to(case.data_type)
    return grad, q, k, v, rab, dense_mask, offsets_q, offsets_k


def _hstu_parameters(case, device):
    context = torch.full((case.batch_size,), case.num_context, dtype=torch.int32, device=device)
    target = torch.full((case.batch_size,), case.num_target, dtype=torch.int32, device=device)
    return context, target


def _build_packed(direction, case, q, k, v, offsets_q, offsets_k, device):
    offsets_q_npu = torch.as_tensor(offsets_q, dtype=torch.int32, device=device)
    offsets_k_npu = torch.as_tensor(offsets_k, dtype=torch.int32, device=device)
    context, target = _hstu_parameters(case, device)
    arbitrary = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(
        offsets_k_npu, context, target, case.target_group_size, int(offsets_k[-1])
    )
    plan_op = torch.ops.ops_rec.get_fwd_attn_plan if direction == "forward" else torch.ops.ops_rec.get_bwd_attn_plan
    tile, sparse, _, _ = plan_op(
        q.to(device),
        k.to(device),
        v.to(device),
        arbitrary,
        offsets_q_npu,
        offsets_k_npu,
        case.max_seq_len,
        case.resolved_max_seq_len_k,
        context,
        target,
        case.target_group_size,
        -1,
        0,
        0,
        True,
    )
    return list(sparse), context, target, list(tile)


def _run_fused(direction, case, pipeline, fused, data, device):
    grad, q, k, v, rab, _, _, _ = data
    metadata = sparse = None
    context = target = None
    if pipeline == PLAN:
        offsets_q_npu = torch.as_tensor(fused.seq_offset_q, dtype=torch.int32, device=device)
        offsets_k_npu = torch.as_tensor(fused.seq_offset_k, dtype=torch.int32, device=device)
        context, target = _hstu_parameters(case, device)
        af = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(
            offsets_k_npu, context, target, case.target_group_size, int(fused.seq_offset_k[-1])
        )
        plan_op = torch.ops.ops_rec.get_fwd_attn_plan if direction == "forward" else torch.ops.ops_rec.get_bwd_attn_plan
        _, sparse, metadata, _ = plan_op(
            q.to(device),
            k.to(device),
            v.to(device),
            af,
            offsets_q_npu,
            offsets_k_npu,
            case.max_seq_len,
            case.resolved_max_seq_len_k,
            context,
            target,
            case.target_group_size,
            -1,
            0,
            0,
            True,
        )
    elif pipeline in (METADATA_ARBITRARY, ARBITRARY):
        sparse, context, target, tile = _build_packed(
            direction, case, q, k, v, fused.seq_offset_q, fused.seq_offset_k, device
        )
        if pipeline == METADATA_ARBITRARY:
            metadata = (
                fused.create_forward_metadata(q, k, v, sparse, tile)
                if direction == "forward"
                else fused.create_backward_metadata(q, k, v, sparse, tile)
            )
    elif pipeline == BUILTIN:
        context, target = _hstu_parameters(case, device)

    if direction == "forward":
        return fused.forward(
            q,
            k,
            v,
            rab,
            None,
            metadata,
            sparse,
            context,
            target,
            case.target_group_size,
            window_size=(-1, 0) if pipeline == PLAN else (-1, -1),
            plan_af=af if pipeline == PLAN else None,
        )
    return fused.backward(
        grad,
        q,
        k,
        v,
        rab,
        None,
        (-1, -1) if pipeline == NO_MASK else (-1, 0),
        context,
        target,
        case.target_group_size,
        metadata,
        sparse,
    )


def run_precision_case(
    direction,
    case,
    pipeline,
    fused_backend,
    native_backend,
    validator,
    device,
    alpha=0.5,
    scale=1 / 1024,
):
    data = _make_case_data(case)
    grad, q, k, v, rab, dense_mask, offsets_q, offsets_k = data
    fused = fused_backend.kernel(
        alpha, scale, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    native = native_backend.kernel(
        alpha, scale, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )
    actual = _run_fused(direction, case, pipeline, fused, data, device)
    if direction == "forward":
        expected = native.forward(q, k, v, rab, None if pipeline == NO_MASK else dense_mask)
        passed, detail = validator.forward_verify(actual, expected)
    else:
        expected = native.backward(grad, q, k, v, rab, None if pipeline == NO_MASK else dense_mask)
        passed, detail = validator.backward_verify(actual, expected)
    del data, fused, native, actual, expected
    torch.npu.empty_cache()
    return passed, detail


def run_benchmark_case(direction, case, pipeline, fused_backend, device, warmup=2, repeats=5):
    data = _make_case_data(case)
    *_, offsets_q, offsets_k = data
    fused = fused_backend.kernel(
        0.5, 1 / 1024, case.has_rab, case.max_seq_len, case.resolved_max_seq_len_k, offsets_q, offsets_k
    )

    def invoke():
        return _run_fused(direction, case, pipeline, fused, data, device)

    for _ in range(warmup):
        invoke()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        invoke()
    end.record()
    torch.npu.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeats
    torch.npu.empty_cache()
    return float(elapsed_ms)


def assert_matrix_coverage(cases):
    cases = list(cases)
    assert {case.batch_size for case in cases} == set(BATCHES)
    assert {case.head_num for case in cases} == set(HEADS)
    assert {case.max_seq_len for case in cases} == set(SEQUENCES)
    assert {case.head_dim for case in cases} == set(HEAD_DIMS)
    assert {case.has_rab for case in cases} == set(RAB_OPTIONS)
    assert any(case.has_rab and case.head_num > 1 and case.rab_heads == 1 for case in cases)
    assert any(case.has_rab and case.head_num > 1 and case.rab_heads is None for case in cases)
    assert sum(case.resolved_max_seq_len_k != case.max_seq_len for case in cases) * 5 >= len(cases) * 3
    assert sum(case.resolved_head_dim_v != case.head_dim for case in cases) * 5 >= len(cases) * 3
    for case in cases:
        if case.resolved_max_seq_len_k != case.max_seq_len and case.batch_size > 1:
            assert min(case.seq_lens) < max(case.seq_lens)
            assert min(case.resolved_seq_lens_k) < max(case.resolved_seq_lens_k)


def assert_ragged_coverage(cases):
    cases = list(cases)
    assert all(case.seq_lens is not None for case in cases)
    assert all(len(case.seq_lens) == case.batch_size for case in cases)
    assert all(min(case.seq_lens) < max(case.seq_lens) for case in cases)
    assert all(all(length % 32 == 0 for length in case.seq_lens) for case in cases)
    assert any(min(case.seq_lens) == 32 and max(case.seq_lens) >= 8192 for case in cases)
    assert {case.head_dim for case in cases} == set(HEAD_DIMS)
    assert {case.has_rab for case in cases} == set(RAB_OPTIONS)
    assert any(case.has_rab and case.head_num > 1 and case.rab_heads == 1 for case in cases)
    assert {case.data_type for case in cases} == set(DTYPES)


def dense_reference_elements(case):
    """Elements of the dense [B, H, Nq, Nk] score matrix the PyTorch golden materializes."""
    return case.batch_size * case.head_num * case.max_seq_len * case.resolved_max_seq_len_k


def user_case_precision_cases():
    """Official user_case shapes expressed as TestCaseParams for precision runs."""
    from utils.user_case_benchmark import SEED, user_cases

    converted = []
    for user_case in user_cases():
        head_dim_v = user_case.dim_v if user_case.dim_v != user_case.dim_qk else None
        max_seq_len_k = user_case.max_k if user_case.max_k != user_case.max_q else None
        converted.append(
            TestCaseParams(
                case_id=user_case.case_id,
                batch_size=user_case.batch,
                head_num=user_case.heads,
                max_seq_len=user_case.max_q,
                head_dim=user_case.dim_qk,
                has_rab=user_case.has_rab,
                data_type=user_case.dtype,
                seed=SEED,
                num_context=0,
                num_target=0,
                target_group_size=1,
                max_seq_len_k=max_seq_len_k,
                head_dim_v=head_dim_v,
                rab_heads=1 if user_case.has_rab else None,
            )
        )
    return converted
