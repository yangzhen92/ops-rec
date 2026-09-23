#!/usr/bin/env python3
"""Build all eleven independent masks with per-batch configuration."""

import torch

import ops_rec


def main():
    device = "npu:0"
    torch.npu.set_device(device)
    offsets = torch.tensor([0, 8, 20], dtype=torch.int32, device=device)
    window_sizes = torch.tensor([3, 7], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([2, 6], dtype=torch.int32, device=device)
    group_sizes = torch.tensor([2, 5], dtype=torch.int32, device=device)
    block_sizes = torch.tensor([4, 6], dtype=torch.int32, device=device)
    valid_lens = torch.tensor([6, 9], dtype=torch.int32, device=device)
    target_lens = torch.tensor([4, 6], dtype=torch.int32, device=device)
    pre_doc_lens = torch.tensor([3, 4], dtype=torch.int32, device=device)
    gist_starts = torch.tensor([5, 14], dtype=torch.int32, device=device)
    gist_ends = torch.tensor([7, 17], dtype=torch.int32, device=device)
    total = int(offsets[-1])
    calls = {
        "causal": lambda: ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(offsets, total),
        "sliding_window": lambda: ops_rec.ascendc.attention.build_sliding_window_mask_arbitrary_func(
            offsets, window_sizes, total),
        "prefix_lm": lambda: ops_rec.ascendc.attention.build_prefix_lm_mask_arbitrary_func(
            offsets, prefix_lens, total),
        "group_wise": lambda: ops_rec.ascendc.attention.build_group_wise_mask_arbitrary_func(
            offsets, group_sizes, total),
        "block_causal": lambda: ops_rec.ascendc.attention.build_block_causal_mask_arbitrary_func(
            offsets, block_sizes, total),
        "grouped_causal": lambda: ops_rec.ascendc.attention.build_grouped_causal_mask_arbitrary_func(
            offsets, group_sizes, total),
        "block_diagonal": lambda: ops_rec.ascendc.attention.build_block_diagonal_mask_arbitrary_func(offsets, total),
        "upper_triangular": lambda: ops_rec.ascendc.attention.build_upper_triangular_mask_arbitrary_func(offsets, total),
        "padding_row": lambda: ops_rec.ascendc.attention.build_padding_row_mask_arbitrary_func(
            offsets, valid_lens, total),
        "target_group_block": lambda: ops_rec.ascendc.attention.build_target_group_block_mask_arbitrary_func(
            offsets, target_lens, group_sizes, total),
        "gist": lambda: ops_rec.ascendc.attention.build_gist_mask_arbitrary_func(
            offsets, pre_doc_lens, gist_starts, gist_ends, total),
    }

    expected_n_func = {
        "target_group_block": 4,
        "gist": 4,
    }
    encoded = {}
    for name, call in calls.items():
        encoded[name] = call()
        assert encoded[name].shape == (1, 1, total, expected_n_func.get(name, 2))
        print(name, encoded[name][0, 0, :3].cpu())

    packed = ops_rec.ascendc.attention.create_fwd_packed_block_sparse(
        encoded["target_group_block"], offsets, offsets, 12, 12, 128, 128
    )
    assert len(packed) == 9
    print("packed descriptor shape:", tuple(packed[6].shape))


if __name__ == "__main__":
    main()
