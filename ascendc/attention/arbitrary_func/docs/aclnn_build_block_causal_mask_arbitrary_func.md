# aclnnBuildBlockCausalMaskArbitraryFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据每个序列的 `offsets` / `block_sizes`，构造 Block-Causal 类型的 arbitrary function 掩码编码（af 编码）。

对每个全局位置 `pos`，二分查找 `offsets` 定位所属序列 `seq_id`，记 `seq_start=offsets[seq_id]`、
`seq_end=offsets[seq_id+1]`、`pos_in_seq=pos-seq_start`、`block_size=block_sizes[seq_id]`（为 0 时退化为 1）。

每个 query 仅能看到自身所在 block 及之前所有 block 的 keys：

- `block_start = floor(pos_in_seq / block_size) * block_size`
- `block_end   = min(block_start + block_size, seq_len)`

输出 `af[pos][0..1]` 两列：

- `col0 = seq_start`           — block 起点的全局索引
- `col1 = seq_start + block_end`  — block 终点（不含）的全局索引

即每个 query 可见的 keys 集合为 `[col0, col1)`（全局坐标，半开区间）。

输出 shape `[1, 1, total_seqlen, 2]`。

### 编码示意图

![build_block_causal_mask_arbitrary_func af编码与注意力mask对应关系](build_block_causal_mask_arbitrary_func示意图.svg)

## 函数原型

两段式接口，必须先调用 `aclnnBuildBlockCausalMaskArbitraryFuncGetWorkspaceSize` 接口获取 workspace 大小和执行器，
再调用 `aclnnBuildBlockCausalMaskArbitraryFunc` 接口执行计算。

```c
aclnnStatus aclnnBuildBlockCausalMaskArbitraryFuncGetWorkspaceSize(
    const aclTensor *offsets,
    const aclTensor *blockSizes,
    int64_t totalSeqLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBuildBlockCausalMaskArbitraryFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnBuildBlockCausalMaskArbitraryFuncGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| offsets（aclTensor） | 输入 | 每序列起始偏移，shape [batch_size+1] | INT32 | ND |
| blockSizes（aclTensor） | 输入 | 每序列的 block 大小，shape [batch_size] | INT32 | ND |
| totalSeqLen（int64_t） | 输入 | 所有序列长度之和，须等于 `offsets` 最后一个元素 | INT64 | 标量 |
| out（aclTensor） | 输出 | af 编码输出，shape [1,1,total_seqlen,2] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | UINT64 | 指针 |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | 执行器指针 | 指针 |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnBuildBlockCausalMaskArbitraryFunc

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnBuildBlockCausalMaskArbitraryFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D，`offsets[0]` 必须为 0、`offsets` 严格单调递增
- `blockSizes` 必须为 1D，shape 须为 `[batch_size]`（`batch_size = offsets.size(0)-1`）
- `totalSeqLen` 须等于 `offsets[-1]`
- `blockSizes` 中各元素须 ≥ 0（为 0 时内部退化为 1）
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_arbitrary_func_block_causal_mask.cpp](../examples/aclnn/test_aclnn_arbitrary_func_block_causal_mask.cpp) 的完整实现。
