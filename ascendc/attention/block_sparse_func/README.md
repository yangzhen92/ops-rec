# compact_no_sync

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |

## 功能说明

flash-attn 中 `compact_block_idx` 的无同步替代实现。给定每个 batch/head/block 的
mask 块数与 full 块数，以及填充后的 `block_idx` 查找表，返回 CSR 风格的 offset 与
压缩后的 index 缓冲，供 `LinearBlockSparseTensors` 使用，避免
`cudaStreamSynchronize` + `.item()` 的同步往返开销。

该算子由两个子算子组合实现：

1. **DualExclusiveScan**：对 mask_cnt / full_cnt 并行做 exclusive scan，得到 mask_offset / full_offset。
2. **ExtractCompact**：根据每行的块数与 offset，从 block_idx 中提取 mask_compact / full_compact，
   其中 full_compact 取每行前 full_cnt 个，mask_compact 取每行最后 mask_cnt 个（逆序）。

对应接口：

- aclnn 接口：`aclnnDualExclusiveScan`、`aclnnExtractCompact`
- torch 接口：`ops_rec.ascendc.attention.compact_no_sync`

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| mask_block_cnt | 输入 | Tensor | INT32 | ND | [B, H, num_blocks] | 每个 query 块的 mask 块数 |
| full_block_cnt | 输入 | Tensor | INT32 | ND | [B, H, num_blocks] | 每个 query 块的 full 块数 |
| block_idx | 输入 | Tensor | INT32 | ND | [B, H, num_blocks, max_blocks] | 块索引查找表 |
| mask_block_cnt_out | 输出 | Tensor | INT32 | ND | [B, H, num_blocks] | 原样返回的 mask 块数 |
| mask_offset | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks+1] | mask 的 exclusive scan offset |
| mask_compact | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks\*max_blocks] | 压缩后的 mask 索引缓冲 |
| full_block_cnt_out | 输出 | Tensor | INT32 | ND | [B, H, num_blocks] | 原样返回的 full 块数 |
| full_offset | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks+1] | full 的 exclusive scan offset |
| full_compact | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks\*max_blocks] | 压缩后的 full 索引缓冲 |

## 约束说明

- 仅支持 INT32 数据类型
- mask_block_cnt、full_block_cnt 必须为 3D 且 shape 相同
- block_idx 必须为 4D，且前三维与 mask_block_cnt/full_block_cnt 一致
- mask_compact、full_compact 尾端可能被过度分配（未初始化），只应读取 offset 末尾之前的有效前缀
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 必须已定义且非空、所有输入 tensor 必须位于同一 NPU 设备。详见 [torch 接口文档](./docs/torch_compact_no_sync.md)。

## 调用示例

- [aclnn DualExclusiveScan 调用示例](./examples/aclnn/test_aclnn_dual_exclusive_scan.cpp)
- [aclnn ExtractCompact 调用示例](./examples/aclnn/test_aclnn_extract_compact.cpp)
- [torch 调用示例](./examples/torch/test_torch_compact_no_sync.py)
- [aclnn DualExclusiveScan 接口文档](./docs/aclnn_dual_exclusive_scan.md)
- [aclnn ExtractCompact 接口文档](./docs/aclnn_extract_compact.md)
- [torch 接口文档](./docs/torch_compact_no_sync.md)
