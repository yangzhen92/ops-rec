# Eleven independent arbitrary-mask operators

These are eleven separate AscendC operators. They do not dispatch through a shared
interval-mask implementation: every mask owns its OpDef, tiling data, kernel,
Torch wrapper, and pytest module. Each output row stores four global-coordinate
half-open intervals as begin/end pairs. Each operator uses the minimum width
needed by its semantics: one interval uses `N_FUNC=2`; target-group-block and
gist use `N_FUNC=4` for at most two intervals.

| Torch API | Per-batch configuration | Visible keys for local query q |
|---|---|---|
| build_causal_mask_arbitrary_func(offsets, total) | sequence lengths from offsets | k <= q |
| build_sliding_window_mask_arbitrary_func(offsets, window_sizes, total) | window_sizes[B] | max(0,q-W+1) <= k <= q |
| build_prefix_lm_mask_arbitrary_func(offsets, prefix_lens, total) | prefix_lens[B] | bidirectional prefix, then causal |
| build_group_wise_mask_arbitrary_func(offsets, group_sizes, total) | group_sizes[B] | same group, bidirectional |
| build_block_causal_mask_arbitrary_func(offsets, block_sizes, total) | block_sizes[B] | current and previous blocks |
| build_grouped_causal_mask_arbitrary_func(offsets, group_sizes, total) | group_sizes[B] | same group and k <= q |
| build_block_diagonal_mask_arbitrary_func(offsets, total) | sequence lengths from offsets | same packed sequence |
| build_upper_triangular_mask_arbitrary_func(offsets, total) | sequence lengths from offsets | k >= q |
| build_padding_row_mask_arbitrary_func(offsets, valid_lens, total) | valid_lens[B] | valid rows to valid keys |
| build_target_group_block_mask_arbitrary_func(offsets, target_lens, group_sizes, total) | both tensors are [B] | causal history plus bidirectional target group |
| build_gist_mask_arbitrary_func(offsets, pre_doc_lens, gist_starts, gist_ends, total) | all config tensors are [B] | pre_doc/doc/gist/query layout; query sees pre_doc + gist + causal query, not raw doc |

All tensors are contiguous NPU int32 tensors. The tests deliberately use a
different configuration for every batch item and validate thirty cases per
operator. Prefix-LM and causal sliding-window semantics follow the public
PyTorch FlexAttention examples. Block-causal follows the usual streaming mask:
tokens in the current block are mutually visible while later blocks are hidden.
Gist-cache follows the SEARCH_ADS layout: `pre_doc=[0,P)`, `doc=[P,G0)`,
`gist=[G0,G1)`, and `query=[G1,S)`. Gist rows absorb doc information; query
rows read the compressed gist representation and do not directly attend to the
raw doc segment.

## Diagrams

- [causal](./build_causal_mask_arbitrary_func示意图.svg)
- [sliding_window](./build_sliding_window_mask_arbitrary_func示意图.svg)
- [prefix_lm](./build_prefix_lm_mask_arbitrary_func示意图.svg)
- [group_wise](./build_group_wise_mask_arbitrary_func示意图.svg)
- [block_causal](./build_block_causal_mask_arbitrary_func示意图.svg)
- [grouped_causal](./build_grouped_causal_mask_arbitrary_func示意图.svg)
- [block_diagonal](./build_block_diagonal_mask_arbitrary_func示意图.svg)
- [upper_triangular](./build_upper_triangular_mask_arbitrary_func示意图.svg)
- [padding_row](./build_padding_row_mask_arbitrary_func示意图.svg)
- [target_group_block](./build_target_group_block_mask_arbitrary_func示意图.svg)
- [gist](./build_gist_mask_arbitrary_func示意图.svg)
