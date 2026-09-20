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

算子在Catlass 样例49_ascend950_flash_attention_infer中修改添加块跳过以及按token的SIMT掩码生成实现任意区间掩码的注意力前向计算。

计算公式：

```text
S = Q @ K^T / sqrt(D)
S = mask_arbitrary(S)          # 只保留 af 区间并集内的位置，其余置 -inf
P = softmax(S, dim=-1)
O = P @ V
```

其中 `mask_arbitrary` 对每个 query 位置 `i` 的 KV 位置 `j` 生效：

```text
valid(i, j) = (af1[i] <= j < af2[i]) || (af3[i] <= j < af4[i])
```

对应接口：

- aclnn 接口：`aclnnFaInferWithArbitraryFunc`
- torch 接口：`ops_rec.ascendc.attention.fa_infer_with_arbitrary_func`

## 算子实现原理

算子基于 CATLASS 与 AscendC 开发，将 QK MatMul、Softmax、PV MatMul 融合在单个 Kernel 内，
并结合块稀疏跳块与按 token 的 SIMT 掩码生成。具体计算步骤如下：

1. 输入 q、k、v（TND 格式，batch 固定为 1），以及函数编码张量 `arbitrary_func`
2. 预处理阶段（算子外完成）：根据 `arbitrary_func` 将每个 (q_block, kv_block) 分类为
   FULL（完全可见）/ MASK（部分可见）/ EMPTY（完全不可见），生成 mask/full 三组块稀疏索引
   （cnt / offset / idx），供算子跳过 EMPTY 块
3. Kernel 内按块遍历 FULL / MASK 的 KV 块并执行 QK MatMul，分数乘以 `1 / sqrt(D)`
4. 对 MASK 块调用 SIMT 线程（`BuildArbitraryMaskSimt`）按 Query token 读取 `arbitrary_func`
   的 4 个区间边界，将不在区间并集内的 KV 位置置为 `MIN_VALUE`；FULL 块直接通过
5. 逐块执行 online softmax，累积 running max 与 exp 和
6. PV MatMul 并对输出 O 执行 rescale（除以当前 exp 和）

Kernel 使用混合 AIC 1:2 架构（`KERNEL_TYPE_MIX_AIC_1_2`），多核并行度取自全部 AIC 核数。

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

> num_rows = ceil(total_seqlen / block_size)，每个 H 共用一个 mask_block_cnt / mask_block_offset / mask_block_idx，
> 其中块大小 block_size=128。
> 通过 torch 接口调用时，q2k_block_sparse 以长度为 6 的 Tensor 列表传入，依次为
> [mask_block_cnt, mask_block_offset, mask_block_idx, full_block_cnt, full_block_offset, full_block_idx]。
> 只支持等长序列，不支持变长序列。
> mask_block_idx 及 full_block_idx 由 compact_no_sync 算子生成，长度一致，实际有效长度分别为 mask_block_offset[-1] 及 full_block_offset[-1]

## 约束说明

- q、k、v 及输出仅支持 BFLOAT16，数据类型需保持一致
- q、k、v 必须为 3D（TND 格式），最后一维（headDim）必须为 128
- batch 固定为 1，多序列需拼接为单个序列后传入；H_q 须为 H_kv 的整数倍（支持 GQA）
- arbitrary_func 必须为 INT32、4D，每 query 固定 4 列（两个区间），行数须与 total_seqlen 对应
- mask 与 block_table 为保留输入，本算子不使用（torch 接口调用时无需传入）
- 块稀疏索引（mask/full 的 cnt/offset/idx）需在算子外由预处理算子或脚本生成，并保证与
  arbitrary_func 语义一致（FULL / MASK / EMPTY 分类正确）
- 算子输入 shape 受显存大小限制
- 该算子需要在 NPU 设备上运行
- 编译依赖 CATLASS 源码，编译前需初始化 submodule：

```bash
git submodule update --init --recursive
```

## 调用示例

- [pytest 功能测试](./fa_infer_with_arbitrary_func_forward/test/pytest/test_fa_infer_with_arbitrary_func.py)
- [torch 接口实现](./torch_extension/fa_infer_with_arbitrary_func_plugin.cpp)

相关算子：

- [build_hstu_mask_arbitrary_func](../arbitrary_func/README.md)：构造 arbitrary function 掩码编码
- [block_sparse_func](../block_sparse_func/README.md)：根据函数编码生成 Q2K/K2Q 块稀疏索引
