# aclnnBuildSlidingWindowMaskArbitraryFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据每个序列的 `offsets` 与 `window_sizes`，构造滑窗掩码（sliding-window mask）的
arbitrary function 编码（af 编码）。

对每个全局位置 `pos`，二分查找 `offsets` 定位所属序列 `seq_id`，写出 `af[pos][0..1]` 两个列
（单个半开区间 `[col0, col1)`，全局坐标）：

- `window = max(window_sizes[seq_id], 1)`
- `col0 = offsets[seq_id] + max(pos_local - window + 1, 0)`（`pos_local = pos - offsets[seq_id]`）
- `col1 = pos + 1`

即 query 位置 `pos` 可见的 key 集合 = `[col0, col1)`：窗口大小为 `window` 的最近几个 key
（本序列内 `key <= pos` 的因果滑窗），窗口外的 key 被掩蔽。

### 编码示意图

![build_sliding_window_mask_arbitrary_func af编码与注意力mask对应关系](build_sliding_window_mask_arbitrary_func示意图.svg)

## 函数原型

两段式接口，必须先调用 `aclnnBuildSlidingWindowMaskArbitraryFuncGetWorkspaceSize` 接口获取 workspace 大小和执行器，
再调用 `aclnnBuildSlidingWindowMaskArbitraryFunc` 接口执行计算。

```c
aclnnStatus aclnnBuildSlidingWindowMaskArbitraryFuncGetWorkspaceSize(
    const aclTensor *offsets,
    const aclTensor *windowSizes,
    int64_t totalSeqLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBuildSlidingWindowMaskArbitraryFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnBuildSlidingWindowMaskArbitraryFuncGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| offsets（aclTensor） | 输入 | 每序列起始偏移，shape [batch_size+1] | INT32 | ND |
| windowSizes（aclTensor） | 输入 | 每序列滑窗大小，shape [batch_size] | INT32 | ND |
| totalSeqLen（int64_t） | 输入 | 所有序列长度之和，须等于 `offsets` 最后一个元素 | INT64 | 标量 |
| out（aclTensor） | 输出 | af 编码输出，shape [1,1,total_seqlen,2] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | UINT64 | 指针 |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | 执行器指针 | 指针 |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnBuildSlidingWindowMaskArbitraryFunc

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnBuildSlidingWindowMaskArbitraryFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D，且维度须满足 `[batch_size+1]`，否则算子执行（tiling 阶段）报错返回
- `window_sizes` 必须为 1D，且维度须满足 `[batch_size]`，否则算子执行（tiling 阶段）报错返回
- `total_seqlen` 须等于 `offsets[-1]`
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_arbitrary_func_sliding_window_mask.cpp](../examples/aclnn/test_aclnn_arbitrary_func_sliding_window_mask.cpp)
的完整实现。示例中 batch=2、offsets=[0,3,5]、window_sizes=[2,1]、total_seqlen=5：
seq0 内 q0→[0,1)、q1→[0,2)、q2→[1,3)；seq1 内 q3→[3,4)、q4→[4,5)。