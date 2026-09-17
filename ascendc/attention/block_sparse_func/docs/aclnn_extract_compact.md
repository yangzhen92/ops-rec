# aclnnExtractCompact

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

## 功能说明

根据每行的 mask/full 块数及其 offset，从 block_idx 查找表中提取出 mask_compact 与
full_compact 两个压缩后的索引缓冲。

- `full_compact` 取每行前 full_cnt 个元素
- `mask_compact` 取每行最后 mask_cnt 个元素（逆序）

mask_offset / full_offset 通常由 `DualExclusiveScan` 算子产生（exclusive scan）。

## 函数原型

两段式接口，必须先调用 `aclnnExtractCompactGetWorkspaceSize` 接口获取 workspace 大小
和执行器，再调用 `aclnnExtractCompact` 接口执行计算。

```c
aclnnStatus aclnnExtractCompactGetWorkspaceSize(
    const aclTensor *blockIdx,
    const aclTensor *maskCnt,
    const aclTensor *fullCnt,
    const aclTensor *maskOffset,
    const aclTensor *fullOffset,
    const aclTensor *maskCompact,
    const aclTensor *fullCompact,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnExtractCompact(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnExtractCompactGetWorkspaceSize

### 参数说明

> 说明：本算子按 maskCnt 的总元素数（展平后的 nBlocks）逐行处理，支持任意前导维度
> （如 torch 路径下 blockIdx 为 4D [B, H, num_blocks, max_blocks]）。下表中 nBlocks
> 表示展平后的块总数，maxBlocks 为 blockIdx 最后一维的长度。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| blockIdx（aclTensor） | 输入 | 块索引查找表，shape [nBlocks, maxBlocks]（可为 4D [B, H, num_blocks, max_blocks]） | INT32 | ND |
| maskCnt（aclTensor） | 输入 | mask 块数，总元素数 nBlocks | INT32 | ND |
| fullCnt（aclTensor） | 输入 | full 块数，总元素数 nBlocks | INT32 | ND |
| maskOffset（aclTensor） | 输入 | mask 的 exclusive scan offset，shape [nBlocks+1] | INT32 | ND |
| fullOffset（aclTensor） | 输入 | full 的 exclusive scan offset，shape [nBlocks+1] | INT32 | ND |
| maskCompact（aclTensor） | 输出 | 压缩后的 mask 索引缓冲，shape [nBlocks*maxBlocks] | INT32 | ND |
| fullCompact（aclTensor） | 输出 | 压缩后的 full 索引缓冲，shape [nBlocks*maxBlocks] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnExtractCompact 接口

### aclnnExtractCompact 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnExtractCompactGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnExtractCompact 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- blockIdx 可为 2D [nBlocks, maxBlocks] 或 4D [B, H, num_blocks, max_blocks]，最后一维为 maxBlocks
- maskCnt、fullCnt 总元素数相等（均等于 nBlocks）
- maskOffset、fullOffset 总元素数等于 nBlocks + 1
- mask_compact、full_compact 尾端可能被过度分配（未初始化），只应读取 offset 末尾之前的有效前缀
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_extract_compact.cpp](../examples/aclnn/test_aclnn_extract_compact.cpp) 的完整实现。
