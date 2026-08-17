# aclnnComputeQBlockKvRange

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |

## 功能说明

为每个 (b, h, q_block) 预计算其覆盖的 kv 区间下界与上界，写入 `q_block_kv_min` 与
`q_block_kv_max`。该算子是 `CreateK2qBlockSparseFromFunc` 的 kv-range 优化前置阶段：
`create_k2q_block_sparse_from_func` 会利用该结果跳过与当前 kv_block 无交叠的 q_block。

`func_tensor` 布局为 [B, H, func_q_len, n_func]，`n_func` 为偶数，每个 q token 的合法
kv 区间为 `[F_{2i}, F_{2i+1})`（i = 0, 1, ..., n_func/2 - 1）的并集。对每个 q_block 内的
所有有效 q token，取其所有合法区间的 min 作为 `q_block_kv_min`，max 作为 `q_block_kv_max`。

## 函数原型

两段式接口，必须先调用 `aclnnComputeQBlockKvRangeGetWorkspaceSize` 接口获取 workspace
大小和执行器，再调用 `aclnnComputeQBlockKvRange` 接口执行计算。

```c
aclnnStatus aclnnComputeQBlockKvRangeGetWorkspaceSize(
    const aclTensor *funcTensor,
    int64_t strideB,
    int64_t strideH,
    int64_t strideQ,
    int64_t strideF,
    int64_t qLen,
    int64_t qBlockSize,
    const aclTensor *qBlockKvMin,
    const aclTensor *qBlockKvMax,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnComputeQBlockKvRange(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnComputeQBlockKvRangeGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| funcTensor（aclTensor） | 输入 | 函数编码张量，shape [B, H, func_q_len, n_func]，n_func 为偶数 | INT32 | ND |
| strideB（int64_t） | 输入 | func_tensor 第 0 维步长 | - | - |
| strideH（int64_t） | 输入 | func_tensor 第 1 维步长 | - | - |
| strideQ（int64_t） | 输入 | func_tensor 第 2 维步长 | - | - |
| strideF（int64_t） | 输入 | func_tensor 第 3 维步长 | - | - |
| qLen（int64_t） | 输入 | Q 序列长度 | - | - |
| qBlockSize（int64_t） | 输入 | Q 块大小 | - | - |
| qBlockKvMin（aclTensor） | 输出 | 每个 q_block 的 kv 区间下界，shape [B, H, num_q_blocks] | INT32 | ND |
| qBlockKvMax（aclTensor） | 输出 | 每个 q_block 的 kv 区间上界，shape [B, H, num_q_blocks] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

> num_q_blocks = ceil(qLen / qBlockSize)。

### 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnComputeQBlockKvRange 接口

### aclnnComputeQBlockKvRange 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnComputeQBlockKvRangeGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnComputeQBlockKvRange 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- func_tensor 必须为 4D，n_func 必须为偶数（如 2、4、6、8、...）
- qLen、qBlockSize 必须大于 0
- qBlockKvMin、qBlockKvMax 的 shape 为 [B, H, num_q_blocks]
- 该算子需要在 NPU 设备上运行

## 调用示例

该算子通常作为 `CreateK2qBlockSparseFromFunc` 的前置阶段被 `create_k2q_block_sparse_from_func`
调用，独立的 aclnn 调用示例参考
[test_aclnn_compute_q_block_kv_range.cpp](../examples/aclnn/test_aclnn_compute_q_block_kv_range.cpp)，
端到端（预计算 + 生成）示例参考
[test_aclnn_create_k2q_block_sparse_from_func.cpp](../examples/aclnn/test_aclnn_create_k2q_block_sparse_from_func.cpp)。
