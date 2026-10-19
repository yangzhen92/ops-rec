# aclnnDualExclusiveScan

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

## 功能说明

对 mask_cnt 与 full_cnt 并行做 exclusive scan，得到 mask_offset 与 full_offset。
两个流相互独立，exclusive scan 即每个位置保存其之前所有元素之和，因此 offset 长度比
输入多 1 且首元素为 0。

## 函数原型

两段式接口，必须先调用 `aclnnDualExclusiveScanGetWorkspaceSize` 接口获取 workspace
大小和执行器，再调用 `aclnnDualExclusiveScan` 接口执行计算。

```c
aclnnStatus aclnnDualExclusiveScanGetWorkspaceSize(
    const aclTensor *maskCnt,
    const aclTensor *fullCnt,
    const aclTensor *maskOffset,
    const aclTensor *fullOffset,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnDualExclusiveScan(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnDualExclusiveScanGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| maskCnt（aclTensor） | 输入 | mask 块数，总元素数 N | INT32 | ND |
| fullCnt（aclTensor） | 输入 | full 块数，总元素数 N | INT32 | ND |
| maskOffset（aclTensor） | 输出 | mask 的 exclusive scan offset，shape [N+1] | INT32 | ND |
| fullOffset（aclTensor） | 输出 | full 的 exclusive scan offset，shape [N+1] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnDualExclusiveScan 接口

### aclnnDualExclusiveScan 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnDualExclusiveScanGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnDualExclusiveScan 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- maskCnt、fullCnt 必须维度相同且元素个数相同（可为 1D [N]，也可为 torch 路径下的 3D [B, H, num_blocks]）
- maskOffset、fullOffset 的 shape 为输入总元素数 + 1
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_dual_exclusive_scan.cpp](../examples/aclnn/test_aclnn_dual_exclusive_scan.cpp) 的完整实现。
