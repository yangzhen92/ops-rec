# fa_infer_with_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

用于 Flash Attention 推理场景的自定义注意力掩码算子。通过函数编码张量 `arbitrary_func`
描述每个 Query token 允许关注的 KV 区间，并配合预先计算好的块稀疏索引跳过无关 KV 块，
实现带任意区间掩码的高效注意力前向计算。

每个 Query token 的合法 KV 位置为两个区间 `[af1, af2)` 与 `[af3, af4)` 的并集，仅位于
区间内的 KV 位置参与注意力计算，其余位置的注意力分数在 softmax 前被置为 `-inf`。

计算公式：

```text
S = Q @ K^T / sqrt(D)
S = mask_arbitrary(S)          # 只保留 af 区间并集内的位置，其余置 -inf
P = softmax(S, dim=-1)
O = P @ V
```

## 接口原型

```python
ops_rec.ascendc.attention.fa_infer_with_arbitrary_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    arbitrary_func: Tensor,
    q2k_block_sparse: List[Tensor],
) -> Tensor
```

`q2k_block_sparse` 为长度为 6 的 Tensor 列表，顺序为
`[mask_block_cnt, mask_block_offset, mask_block_idx, full_block_cnt, full_block_offset, full_block_idx]`，
由预处理阶段根据 `arbitrary_func` 生成。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| q | 输入 | Tensor | BFLOAT16 | ND | [total_seqlen, H, D] | Query，TND 格式，D 固定为 128 |
| k | 输入 | Tensor | BFLOAT16 | ND | [total_seqlen, H, D] | Key，数据类型与 q 保持一致 |
| v | 输入 | Tensor | BFLOAT16 | ND | [total_seqlen, H, D] | Value，数据类型与 q 保持一致 |
| arbitrary_func | 输入 | Tensor | INT32 | ND | [1, 1, total_seqlen, 4] | 函数编码张量，每个 query 用 4 个 INT32 表示两个合法 KV 区间 [af1, af2) 与 [af3, af4) |
| mask_block_cnt | 输入 | Tensor | INT32 | ND | [1, 1, num_rows] | 每个 (b, h, q_block) 的 MASK 块个数 |
| mask_block_offset | 输入 | Tensor | INT32 | ND | [num_rows + 1] | mask 块索引的 exclusive scan 偏移 |
| mask_block_idx | 输入 | Tensor | INT32 | ND | [num_mask_block] | 压缩后的 MASK 块编号 |
| full_block_cnt | 输入 | Tensor | INT32 | ND | [1, 1, num_rows] | 每个 (b, h, q_block) 的 FULL 块个数 |
| full_block_offset | 输入 | Tensor | INT32 | ND | [num_rows + 1] | full 块索引的 exclusive scan 偏移 |
| full_block_idx | 输入 | Tensor | INT32 | ND | [num_full_block] | 压缩后的 FULL 块编号 |
| out | 输出 | Tensor | BFLOAT16 | ND | [total_seqlen, H, D] | 输出，shape 与数据类型均与 q 保持一致 |

> num_rows = ceil(total_seqlen / 128)，每个 H 共用一个 mask_block_cnt / mask_block_offset / mask_block_idx。
> 只支持等长序列，不支持变长序列。
> mask_block_idx 及 full_block_idx 由 compact_no_sync 算子生成，长度一致，实际有效长度分别为 mask_block_offset[-1] 及 full_block_offset[-1]

## 约束说明

约束分为两层：算子自身语义适用于 AscendCL 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.attention.fa_infer_with_arbitrary_func` 时生效。

### 算子约束

- q、k、v 及输出仅支持 BFLOAT16，数据类型需保持一致
- q、k、v 必须为 3D（TND 格式），最后一维（headDim）必须为 128
- batch 固定为 1，多序列需拼接为单个序列后传入；
- arbitrary_func 必须为 INT32、4D，每 query 固定 4 列（两个区间），行数须与 total_seqlen 对应
- mask 与 block_table 为保留输入，本算子不使用，通过 torch 接口调用时无需传入
- 块稀疏索引（mask/full 的 cnt/offset/idx）需在算子外由预处理算子或脚本生成，并保证与
  arbitrary_func 语义一致（FULL / MASK / EMPTY 分类正确）
- 算子输入 shape 受显存大小限制
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [fa_infer_with_arbitrary_func_plugin.cpp](../../torch_extension/fa_infer_with_arbitrary_func_plugin.cpp) 在调用算子前完成：

- q、k、v 必须为 3D（TND 格式）
- q、k、v 的最后一维（headDim）必须为 128
- arbitrary_func 必须为 4D
- q2k_block_sparse 长度必须为 6，顺序为 [mask_block_cnt, mask_block_offset, mask_block_idx, full_block_cnt, full_block_offset, full_block_idx]
- 输入 tensor 必须位于同一 NPU 设备
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_fa_infer_with_arbitrary_func.py](../examples/torch/test_torch_fa_infer_with_arbitrary_func.py)。
