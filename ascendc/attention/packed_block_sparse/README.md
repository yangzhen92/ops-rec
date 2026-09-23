# packed_block_sparse

`packed_block_sparse` converts an `arbitrary_func` tensor directly into the
block descriptors and 128-bit row masks consumed by HSTU V2. It is a separate
vendor and does not change the existing `block_sparse_func` API.

## Layout

Both Q2K and K2Q APIs return a nine-tensor list compatible with HSTU V2:

1. `mask_block_cnt [B,H,outer]`
2. `row_bounds [B,H,outer,2]`, inclusive first/last non-empty block
3. reserved one-element placeholder (HSTU metadata is produced by AICPU)
4. `full_block_cnt [B,H,outer]`
5. reserved placeholder
6. reserved placeholder
7. `direct_descriptor [B,H,outer,inner]`
8. one-element placeholder marking direct-descriptor mode for HSTU V2
9. `partial_bitmap [B*H*outer,inner,q_block,kv_block/32]`

Each descriptor stores `(inner_block << 2) | state`, where the low two bits are
`00 EMPTY`, `01 FULL`, or `11 PARTIAL`. HSTU V2 recognizes tensor 8 as a
one-dimensional placeholder and consumes tensor 7 as this direct descriptor, so
there is no `FinalizePackedBlockSparse` pass. FULL blocks skip bitmap reads and
EMPTY blocks skip compute. Each PARTIAL bitmap row contains four consecutive
`int32` words for a 128-column tile or two words for a 64-column tile.

## Torch API

```python
q2k = torch.ops.ops_rec.create_fwd_packed_block_sparse(
    arbitrary_func, q_offsets, kv_offsets, max_seqlen_q, max_seqlen_kv, 128, 128)
k2q = torch.ops.ops_rec.create_bwd_packed_block_sparse(
    arbitrary_func, q_offsets, kv_offsets, max_seqlen_q, max_seqlen_kv, 256, 128)
```

For HSTU head dimensions above 128, use `(q_block, kv_block)=(64,64)`
for Q2K and `(128,64)` for K2Q, matching the HSTU V2 tiling policy.

`arbitrary_func` uses the global variable-length layout produced by
`build_hstu_mask_arbitrary_func`: `[1, Hmask, total_seqlen, 2*groups]`
(or the equivalent `[total_seqlen, 2*groups]` for one mask head).
`q_offsets` and `kv_offsets` are NPU int32 tensors of shape `[batch_size + 1]`.
The AF T axis follows `kv_offsets`; when Q is shorter than KV the producer uses
the last Q AF rows of that sequence. It emits the existing padded rectangular
sparse-info ABI using the two max sequence lengths; callers do not build or
copy a `[B, Hmask, max_seqlen, 2*groups]` AF tensor. The current varlen API is
for right-aligned HSTU attention and requires each `q_len <= kv_len`.

Python list entries 2, 4, and 5 remain lightweight placeholders; entry 7 is the
one-element direct-mode marker. Scheduling metadata is generated exclusively by
the `atten_metadata` AICPU operator; packed sparse data is the
direct descriptor/count/bitmap payload produced by Create*.

The implementation uses one AIV block per outer row and eight warps to scan
inner blocks. Q2K caches each row's arbitrary-function intervals in registers.
K2Q and Q2K both emit row bounds in the same pass for HSTU scheduling and
metadata load balancing.
