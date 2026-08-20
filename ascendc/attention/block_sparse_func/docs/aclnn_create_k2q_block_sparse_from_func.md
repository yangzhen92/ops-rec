# aclnnCreateK2qBlockSparseFromFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |

## 功能说明

K2Q（反向/transpose）：根据函数编码张量 `func_tensor`，为每个 (b, h, kv_block) 并行
计算其覆盖的 q_block 列表，输出块稀疏张量（mask 块数、full 块数、块索引表）。

- 每个 (q_block, kv_block) 块按 `func_tensor` 中的合法 kv 区间分类为：
  - **FULL**：块内所有 (q_token, kv 位置) 均合法
  - **MASK**：块内部分位置合法
  - **EMPTY**：块内无合法位置
- `full_block_cnt` 记录 FULL 块数，`mask_block_cnt` 记录 MASK 块数
- `block_idx` 每行内：FULL 块 id 从左往右写，MASK 块 id 从右往左写（其余位置为 -1）
- K2Q 始终进行 q_boundary 检查：部分 q_block（不足 q_block_size）不能判为 FULL

`func_tensor` 布局为 [B, H, func_q_len, n_func]，`n_func` 为偶数，每个 q token 的合法
kv 区间为 `[F_{2i}, F_{2i+1})` 的并集。该算子可传入 `ComputeQBlockKvRange` 预计算得到的
`q_block_kv_min` / `q_block_kv_max`，用于跳过与当前 kv_block 无交叠的 q_block（kv-range
优化）。

## 函数原型

两段式接口，必须先调用 `aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize` 接口获取
workspace 大小和执行器，再调用 `aclnnCreateK2qBlockSparseFromFunc` 接口执行计算。

```c
aclnnStatus aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize(
    const aclTensor *funcTensor,
    const aclTensor *qBlockKvMin,
    const aclTensor *qBlockKvMax,
    int64_t strideB,
    int64_t strideH,
    int64_t strideQ,
    int64_t strideF,
    int64_t qLen,
    int64_t kvLen,
    int64_t qBlockSize,
    int64_t kvBlockSize,
    const aclTensor *maskBlockCnt,
    const aclTensor *fullBlockCnt,
    const aclTensor *blockIdx,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnCreateK2qBlockSparseFromFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize

### 参数说明

> 说明：num_q_blocks = ceil(qLen / qBlockSize)，num_kv_blocks = ceil(kvLen / kvBlockSize)。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| funcTensor（aclTensor） | 输入 | 函数编码张量，shape [B, H, func_q_len, n_func]，n_func 为偶数 | INT32 | ND |
| qBlockKvMin（aclTensor） | 输入 | 每个 q_block 的 kv 区间下界，shape [B, H, num_q_blocks]，由 ComputeQBlockKvRange 产生 | INT32 | ND |
| qBlockKvMax（aclTensor） | 输入 | 每个 q_block 的 kv 区间上界，shape [B, H, num_q_blocks]，由 ComputeQBlockKvRange 产生 | INT32 | ND |
| strideB（int64_t） | 输入 | func_tensor 第 0 维步长 | - | - |
| strideH（int64_t） | 输入 | func_tensor 第 1 维步长 | - | - |
| strideQ（int64_t） | 输入 | func_tensor 第 2 维步长 | - | - |
| strideF（int64_t） | 输入 | func_tensor 第 3 维步长 | - | - |
| qLen（int64_t） | 输入 | Q 序列长度 | - | - |
| kvLen（int64_t） | 输入 | KV 序列长度 | - | - |
| qBlockSize（int64_t） | 输入 | Q 块大小 | - | - |
| kvBlockSize（int64_t） | 输入 | KV 块大小 | - | - |
| maskBlockCnt（aclTensor） | 输出 | 每个 kv_block 的 mask 块数，shape [B, H, num_kv_blocks] | INT32 | ND |
| fullBlockCnt（aclTensor） | 输出 | 每个 kv_block 的 full 块数，shape [B, H, num_kv_blocks] | INT32 | ND |
| blockIdx（aclTensor） | 输出 | 块索引表，shape [B, H, num_kv_blocks, num_q_blocks] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnCreateK2qBlockSparseFromFunc 接口

### aclnnCreateK2qBlockSparseFromFunc 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnCreateK2qBlockSparseFromFunc 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- func_tensor 必须为 4D，n_func 必须为偶数（如 2、4、6、8、...）
- qLen、kvLen、qBlockSize、kvBlockSize 必须大于 0
- qBlockKvMin、qBlockKvMax 必须与 func_tensor 的 B、H 一致，shape 为 [B, H, num_q_blocks]
- block_idx 每行尾端可能被过度分配（未初始化），只应读取前full_block_cnt个FULL块及后mask_block_cnt个MASK块的有效数据
- 该算子需要在 NPU 设备上运行

## 调用示例

完整调用示例参考 [test_aclnn_block_sparse_func_create_k2q_block_sparse_from_func.cpp](../examples/aclnn/test_aclnn_block_sparse_func_create_k2q_block_sparse_from_func.cpp)
（含 `ComputeQBlockKvRange` 预计算阶段），或参考 [torch 接口文档](./torch_create_k2q_block_sparse_from_func.md)
与对应的 [torch 示例](../examples/torch/test_torch_block_sparse_func_create_k2q_block_sparse_from_func.py)。
