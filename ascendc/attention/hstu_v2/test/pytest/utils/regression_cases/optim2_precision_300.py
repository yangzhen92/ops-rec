"""Optim2 branch-coverage matrix with exactly 300 precision cases."""

import os

import pytest
import torch

from utils.feature_matrix import (
    ARBITRARY,
    BUILTIN,
    METADATA_ARBITRARY,
    NO_MASK,
    TestCaseParams as CaseParams,
    create_backends,
    run_precision_case,
)

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))
TOTAL_CASES = 300
LONG_CASES = 30
SHARD_COUNT = int(os.getenv("HSTU_OPTIM2_SHARD_COUNT", "1"))
SHARD_INDEX = int(os.getenv("HSTU_OPTIM2_SHARD_INDEX", "0"))
CaseParams.__test__ = False


def _make_cases():
    cases = []
    short_lengths = (32, 64, 96, 128, 160, 192, 256, 288, 512, 1024)
    dims = (32, 64, 128, 256)
    pipelines = (METADATA_ARBITRARY, ARBITRARY, BUILTIN, NO_MASK)
    for index in range(TOTAL_CASES):
        is_long = index % 10 == 0
        max_seq_len = 4096 if is_long else short_lengths[(index * 7 + index // 10) % len(short_lengths)]
        if is_long or max_seq_len >= 1024:
            batch_size, head_num = 1, 1 + (index % 2)
        else:
            batch_size = (1, 2, 4, 8)[(index // 3) % 4]
            head_num = (1, 2, 4)[(index // 5) % 3]
        direction = "forward" if index % 2 == 0 else "backward"
        pipeline = pipelines[(index // 2) % len(pipelines)]
        dtype = torch.float16 if (index // 8) % 2 == 0 else torch.bfloat16
        has_rab = (index // 16) % 2 == 1 and not is_long
        dim = dims[(index // 4) % len(dims)]
        mask_variant = (index // 32) % 4
        context = max_seq_len // 8 if pipeline != NO_MASK and mask_variant in (1, 3) else 0
        target = max_seq_len // 8 if pipeline != NO_MASK and mask_variant in (2, 3) else 0
        group = (1, 2, 4, 8, 16)[index % 5]
        seq_lens = None
        if batch_size > 1 and index % 3 != 0:
            aligned = []
            for batch in range(batch_size):
                steps = 1 + ((index + batch * 3) % max(1, max_seq_len // 32))
                aligned.append(min(max_seq_len, steps * 32))
            aligned[-1] = max_seq_len
            seq_lens = tuple(aligned)
        min_seq_len = min(seq_lens) if seq_lens is not None else max_seq_len
        if context and target:
            per_region = max(1, (min_seq_len - 1) // 2)
            context = min(context, per_region)
            target = min(target, per_region)
        elif context:
            context = min(context, max(1, min_seq_len - 1))
        elif target:
            target = min(target, max(1, min_seq_len - 1))
        case = CaseParams(
            case_id=index + 1,
            batch_size=batch_size,
            head_num=head_num,
            max_seq_len=max_seq_len,
            head_dim=dim,
            has_rab=has_rab,
            data_type=dtype,
            seed=2026092500 + index,
            num_context=context,
            num_target=target,
            target_group_size=group,
            seq_lens=seq_lens,
        )
        cases.append((direction, pipeline, case))
    return cases


ALL_CASES = _make_cases()
assert len(ALL_CASES) == TOTAL_CASES
assert sum(case.max_seq_len >= 4096 for _, _, case in ALL_CASES) == LONG_CASES
assert {direction for direction, _, _ in ALL_CASES} == {"forward", "backward"}
assert {pipeline for _, pipeline, _ in ALL_CASES} == {METADATA_ARBITRARY, ARBITRARY, BUILTIN, NO_MASK}
assert {case.head_dim for _, _, case in ALL_CASES} == {32, 64, 128, 256}
assert {case.has_rab for _, _, case in ALL_CASES} == {False, True}

SHARD_CASES = [item for index, item in enumerate(ALL_CASES) if index % SHARD_COUNT == SHARD_INDEX]


@pytest.fixture(scope="session")
def backends():
    torch.npu.set_device(DEVICE_ID)
    return create_backends(DEVICE_ID)


@pytest.mark.parametrize(
    "direction,pipeline,case",
    SHARD_CASES,
    ids=lambda value: (
        f"optim2-{value.case_id:03d}-b{value.batch_size}-h{value.head_num}-"
        f"s{value.max_seq_len}-d{value.head_dim}-{value.dtype_name}-rab{int(value.has_rab)}"
        if isinstance(value, CaseParams)
        else str(value)
    ),
)
def test_optim2_precision_300(direction, pipeline, case, backends):
    fused, native, validator = backends
    passed, detail = run_precision_case(direction, case, pipeline, fused, native, validator, f"npu:{DEVICE_ID}")
    assert passed, detail
