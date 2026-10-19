# block_sparse_func

## compact_no_sync算子

### compact_no_sync产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

### compact_no_sync功能说明

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

### compact_no_sync参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| mask_block_cnt | 输入 | Tensor | INT32 | ND | [B, H, num_blocks] | 每个 query 块的 mask 块数 |
| full_block_cnt | 输入 | Tensor | INT32 | ND | [B, H, num_blocks] | 每个 query 块的 full 块数 |
| block_idx | 输入 | Tensor | INT32 | ND | [B, H, num_blocks, max_blocks] | 块索引查找表 |
| mask_block_cnt_out | 输出 | Tensor | INT32 | ND | [B, H, num_blocks] | 原样返回的 mask 块数 |
| mask_offset | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks+1] | mask 的 exclusive scan offset |
| mask_compact | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks\*max_blocks] | 压缩后的 mask 索引缓冲 |
| full_block_cnt_out | 输出 | Tensor | INT32 | ND | [B, H, num_blocks] | 原样返回的 full 块数 |
| full_offset | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks+1] | full 的 exclusive scan offset |
| full_compact | 输出 | Tensor | INT32 | ND | [B\*H\*num_blocks\*max_blocks] | 压缩后的 full 索引缓冲 |

### 约束说明

- 仅支持 INT32 数据类型
- mask_block_cnt、full_block_cnt 必须为 3D 且 shape 相同
- block_idx 必须为 4D，且前三维与 mask_block_cnt/full_block_cnt 一致
- mask_compact、full_compact 尾端可能被过度分配（未初始化），只应读取 offset 末尾之前的有效前缀
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 必须已定义且非空、所有输入 tensor 必须位于同一 NPU 设备。详见 [torch 接口文档](./docs/torch_compact_no_sync.md)。

### 调用示例

- [aclnn DualExclusiveScan 调用示例](./examples/aclnn/test_aclnn_dual_exclusive_scan.cpp)
- [aclnn ExtractCompact 调用示例](./examples/aclnn/test_aclnn_extract_compact.cpp)
- [torch 调用示例](./examples/torch/test_torch_compact_no_sync.py)
- [aclnn DualExclusiveScan 接口文档](./docs/aclnn_dual_exclusive_scan.md)
- [aclnn ExtractCompact 接口文档](./docs/aclnn_extract_compact.md)
- [torch 接口文档](./docs/torch_compact_no_sync.md)

## create_q2k_block_sparse_from_func算子

### create_q2k_block_sparse_from_func产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

### create_q2k_block_sparse_from_func功能说明

Q2K（正向）块稀疏张量生成：根据函数编码张量 `func_tensor`，为每个 (b, h, q_block)
并行计算其覆盖的 kv_block 列表，输出 mask 块数、full 块数与块索引表，供
`LinearBlockSparseTensors` / FlexAttention block-mask 使用。

- 每个 (q_block, kv_block) 块按 `func_tensor` 中的合法 kv 区间分类为 FULL / MASK / EMPTY
- `full_block_cnt` 记录 FULL 块数，`mask_block_cnt` 记录 MASK 块数
- `block_idx` 每行内：FULL 块 id 从左往右写，MASK 块 id 从右往左写（其余位置为 -1）
- `check_q_boundary` 为 true 时，部分 q_block（不足 q_block_size）不能判为 FULL

`func_tensor` 布局为 [B, H, func_q_len, n_func]，`n_func` 为偶数，每个 q token 的合法
kv 区间为 `[F_{2i}, F_{2i+1})` 的并集。该算子为 Q2K 正向路径，对应的反向路径见
`create_k2q_block_sparse_from_func`（输出维度按 kv_block 展开）。

对应接口：

- aclnn 接口：`aclnnCreateQ2kBlockSparseFromFunc`
- torch 接口：`ops_rec.ascendc.attention.create_q2k_block_sparse_from_func`

### create_q2k_block_sparse_from_func参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| func_tensor | 输入 | Tensor | INT32 | ND | [B, H, func_q_len, n_func] | 函数编码张量，n_func 为偶数 |
| q_len | 输入 | int | - | - | - | Q 序列长度 |
| kv_len | 输入 | int | - | - | - | KV 序列长度 |
| q_block_size | 输入 | int | - | - | - | Q 块大小 |
| kv_block_size | 输入 | int | - | - | - | KV 块大小 |
| check_q_boundary | 输入 | bool | - | - | - | 部分 q_block 是否不允许 FULL（q_boundary 检查） |
| debug | 输入 | bool | - | - | - | 调试用，true 时输出初始化为 -1 |
| mask_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks] | 每个 q_block 的 mask 块数 |
| full_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks] | 每个 q_block 的 full 块数 |
| block_idx | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks, num_kv_blocks] | 块索引表 |

> num_q_blocks = ceil(q_len / q_block_size)，num_kv_blocks = ceil(kv_len / kv_block_size)。

### create_q2k_block_sparse_from_func约束说明

- 仅支持 INT32 数据类型
- func_tensor 必须为 4D，n_func 必须为偶数（如 2、4、6、8、...）
- func_q_len 必须大于等于 q_len
- q_len、kv_len、q_block_size、kv_block_size 必须大于 0
- q_block_size、kv_block_size 必须是32的倍数
- block_idx 每行尾端可能被过度分配（未初始化），只应读取前full_block_cnt个FULL块及后mask_block_cnt个MASK块的有效数据
- 该算子需要在 NPU 设备上运行

### create_q2k_block_sparse_from_func调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_block_sparse_func_create_q2k_block_sparse_from_func.cpp)
- [torch 调用示例](./examples/torch/test_torch_block_sparse_func_create_q2k_block_sparse_from_func.py)
- [aclnn 接口文档](./docs/aclnn_create_q2k_block_sparse_from_func.md)
- [torch 接口文档](./docs/torch_create_q2k_block_sparse_from_func.md)

## create_k2q_block_sparse_from_func算子

### create_k2q_block_sparse_from_func产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

### create_k2q_block_sparse_from_func功能说明

K2Q（反向/transpose）块稀疏张量生成：根据函数编码张量 `func_tensor`，为每个
(b, h, kv_block) 计算其覆盖的 q_block 列表，输出 mask 块数、full 块数与块索引表，
供 `LinearBlockSparseTensors` / FlexAttention block-mask 使用。

- 每个 (q_block, kv_block) 块按 `func_tensor` 中的合法 kv 区间分类为 FULL / MASK / EMPTY
- `full_block_cnt` 记录 FULL 块数，`mask_block_cnt` 记录 MASK 块数
- `block_idx` 每行内：FULL 块 id 从左往右写，MASK 块 id 从右往左写（其余位置为 -1）
- K2Q 始终进行 q_boundary 检查：部分 q_block（不足 q_block_size）不能判为 FULL

`func_tensor` 布局为 [B, H, func_q_len, n_func]，`n_func` 为偶数，每个 q token 的合法
kv 区间为 `[F_{2i}, F_{2i+1})` 的并集。

内部实现按任务规模自适应选择路径：

- 大 shape（两阶段）：第一阶段以 (b, h, q_block, kv 列 tile) 为任务粒度（寄存器缓存 func 值，
  消除旧版按 kv_block 切分导致的 func_tensor 读放大；大 q_len、小 B*H 用例行任务不足时，
  由 Host 按 2 倍核数目标把每行 kv 列切成多个 tile 倍增 warp 并行度，避免少量核激活、
  每 warp 串行扫整行 kv 列），将块状态写入 workspace 状态矩阵；
  跨核同步后第二阶段按列有序紧缩，保证 full 列表递增、mask 列表递减
- 小 shape（单阶段）：当并行度不足且 func_tensor 列访问量不超过阈值（按 L2 流量
  预算 128MB）时，以 (b, h, 连续 kv 列组) 为任务，block 内 8 个 warp 交错各领一列、
  warp 级 ballot 归约（无块内栅栏）单遍有序写出结果，省去状态矩阵与跨核同步开销；
  列组宽度 `kvColsPerTask` 由 Host 按 2 倍核数目标任务数动态设定（8 的倍数，[8, 64]），
  兼顾"多 block 小规模"（q_len 小、kv_block_size 小、num_kv_blocks 100+）场景的
  满核并行与任务调度开销
- 每 q_block 的 kv 遍历范围由 kernel 内 warp/block 归约计算（kv-range 裁剪）

对应接口：

- aclnn 接口：`aclnnCreateK2qBlockSparseFromFunc`
- torch 接口：`ops_rec.ascendc.attention.create_k2q_block_sparse_from_func`

### create_k2q_block_sparse_from_func参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| func_tensor | 输入 | Tensor | INT32 | ND | [B, H, func_q_len, n_func] | 函数编码张量，n_func 为偶数 |
| q_len | 输入 | int | - | - | - | Q 序列长度 |
| kv_len | 输入 | int | - | - | - | KV 序列长度 |
| q_block_size | 输入 | int | - | - | - | Q 块大小 |
| kv_block_size | 输入 | int | - | - | - | KV 块大小 |
| debug | 输入 | bool | - | - | - | 调试用，true 时输出初始化为 -1 |
| mask_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_kv_blocks] | 每个 kv_block 的 mask 块数 |
| full_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_kv_blocks] | 每个 kv_block 的 full 块数 |
| block_idx | 输出 | Tensor | INT32 | ND | [B, H, num_kv_blocks, num_q_blocks] | 块索引表 |

> num_q_blocks = ceil(q_len / q_block_size)，num_kv_blocks = ceil(kv_len / kv_block_size)。

### create_k2q_block_sparse_from_func约束说明

- 仅支持 INT32 数据类型
- func_tensor 必须为 4D，n_func 必须为偶数（如 2、4、6、8、...）
- func_q_len 必须大于等于 q_len
- q_len、kv_len、q_block_size、kv_block_size 必须大于 0
- q_block_size、kv_block_size 必须是32的倍数
- block_idx 每行尾端可能被过度分配（未初始化），只应读取前full_block_cnt个FULL块及后mask_block_cnt个MASK块的有效数据
- 该算子需要在 NPU 设备上运行

### create_k2q_block_sparse_from_func调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_block_sparse_func_create_k2q_block_sparse_from_func.cpp)
- [torch 调用示例](./examples/torch/test_torch_block_sparse_func_create_k2q_block_sparse_from_func.py)
- [aclnn 接口文档](./docs/aclnn_create_k2q_block_sparse_from_func.md)
- [torch 接口文档](./docs/torch_create_k2q_block_sparse_from_func.md)
