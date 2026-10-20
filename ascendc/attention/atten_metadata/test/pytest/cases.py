#!/usr/bin/env python3
"""Deterministic 100-case matrices shared by task2 tests and benchmarks."""

import random


def _lengths(case_id, batch, max_seq_len):
    rng = random.Random(20260920 + case_id * 97 + batch * 13 + max_seq_len)
    floor = max(32, max_seq_len // 4)
    values = [rng.randint(floor, max_seq_len) for _ in range(batch)]
    values[case_id % batch] = max_seq_len
    return values


def operator_cases():
    batches = (1, 2, 4, 8, 16)
    heads = (1, 2, 4, 8)
    lengths = (128, 256, 512, 1024, 2048)
    rows = []
    case_id = 0
    for batch in batches:
        for head in heads:
            for max_seq_len in lengths:
                case_id += 1
                seq_lens = _lengths(case_id, batch, max_seq_len)
                offsets = [0]
                for length in seq_lens:
                    offsets.append(offsets[-1] + length)
                rows.append({
                    "case_id": case_id,
                    "batch": batch,
                    "heads": head,
                    "max_seq_len": max_seq_len,
                    "seq_lens": seq_lens,
                    "offsets": offsets,
                    "head_dim": (32, 64, 128)[(case_id - 1) % 3],
                    "has_arbitrary": case_id % 2 == 0,
                    "direction": "q2k" if case_id % 2 else "k2q",
                    "seed": 20260920 + case_id,
                })
    assert len(rows) == 100
    return rows

