# aclnnFaInferWithArbitraryFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | × |
| Atlas A2  训练/推理系列产品 | × |

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

## 函数原型

两段式接口，必须先调用 `aclnnFaInferWithArbitraryFuncGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnFaInferWithArbitraryFunc` 接口执行计算。

```c
aclnnStatus aclnnFaInferWithArbitraryFuncGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *mask,
    const aclTensor *blockTable,
    const aclTensor *arbitraryFunc,
    const aclTensor *maskBlockCnt,
    const aclTensor *maskBlockOffset,
    const aclTensor *maskBlockIdx,
    const aclTensor *fullBlockCnt,
    const aclTensor *fullBlockOffset,
    const aclTensor *fullBlockIdx,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFaInferWithArbitraryFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnFaInferWithArbitraryFuncGetWorkspaceSize 接口

### aclnnFaInferWithArbitraryFuncGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| q（aclTensor） | 输入 | Query，shape [s_q, H_q, D]，TND 格式，D 固定为 128 | FLOAT16 / BFLOAT16 | ND |
| k（aclTensor） | 输入 | Key，shape [s_kv, H_kv, D]，数据类型与 q 保持一致 | FLOAT16 / BFLOAT16 | ND |
| v（aclTensor） | 输入 | Value，shape [s_kv, H_kv, D]，数据类型与 q 保持一致 | FLOAT16 / BFLOAT16 | ND |
| mask（aclTensor） | 输入 | 保留输入，本算子不使用，传 nullptr | - | - |
| blockTable（aclTensor） | 输入 | 保留输入，本算子不使用，传 nullptr | - | - |
| arbitraryFunc（aclTensor） | 输入 | 函数编码张量，shape [1, 1, s_q, 4]，每个 query 用 4 个 INT32 表示两个合法 KV 区间 | INT32 | ND |
| maskBlockCnt（aclTensor） | 输入 | 每个 (b, h, q_block) 的 MASK 块个数，shape [1, 1, num_rows] | INT32 | ND |
| maskBlockOffset（aclTensor） | 输入 | mask 块索引的 exclusive scan 偏移，shape [num_rows + 1] | INT32 | ND |
| maskBlockIdx（aclTensor） | 输入 | 压缩后的 MASK 块编号（-1 已剔除），shape [num_mask_block] | INT32 | ND |
| fullBlockCnt（aclTensor） | 输入 | 每个 (b, h, q_block) 的 FULL 块个数，shape [1, 1, num_rows] | INT32 | ND |
| fullBlockOffset（aclTensor） | 输入 | full 块索引的 exclusive scan 偏移，shape [num_rows + 1] | INT32 | ND |
| fullBlockIdx（aclTensor） | 输入 | 压缩后的 FULL 块编号（-1 已剔除），shape [num_full_block] | INT32 | ND |
| out（aclTensor） | 输出 | 输出，shape [s_q, H_q, D]，数据类型与 q 保持一致 | FLOAT16 / BFLOAT16 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

> num_rows = ceil(s_q / 128)，每个 H_q 共用一个 mask_block_cnt / mask_block_offset / mask_block_idx。

### aclnnFaInferWithArbitraryFuncGetWorkspaceSize 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnFaInferWithArbitraryFunc 接口

### aclnnFaInferWithArbitraryFunc 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnFaInferWithArbitraryFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnFaInferWithArbitraryFunc 返回值

`aclnnStatus`：返回状态码，成功返回 `ACLNN_SUCCESS`。

## aclnnFaInferWithArbitraryFunc 约束说明

- 仅支持 Atlas 950 系列产品
- q、k、v 及输出仅支持 FLOAT16 / BFLOAT16，数据类型需保持一致
- q、k、v 必须为 3D（TND 格式），最后一维（headDim）必须为 128
- batch 固定为 1，多序列需拼接为单个序列后传入；H_q 须为 H_kv 的整数倍（支持 GQA）
- arbitraryFunc 必须为 INT32、4D，每 query 固定 4 列（两个区间），行数须与 s_q 对应
- mask 与 blockTable 为保留输入，本算子不使用，传 nullptr 即可
- 块稀疏索引（mask/full 的 cnt/offset/idx）需在算子外由预处理算子或脚本生成，并保证与
  arbitraryFunc 语义一致（FULL / MASK / EMPTY 分类正确）
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_fa_infer_with_arbitrary_func.cpp](../examples/aclnn/test_aclnn_fa_infer_with_arbitrary_func.cpp) 的完整实现。
