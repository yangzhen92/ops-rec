# aclnnIndexSelectForRank1Backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

`gather_for_rank1`（1D 嵌入权重表按 index 查表，`y = x[index]`）的反向算子：把每个位置的上游梯度 `grad_y[i]` 按 `index[i]` 散射累加回权重表 grad_x（scatter-add），grad_x 与权重表 x 同形。

计算公式：

```text
grad_x[j] = sum(grad_y[i])    (对所有满足 index[i] == j 的 i 求和)
```

> kernel 通过 AtomicAdd 向 grad_x 逐核累加，调用前 grad_x 必须已清零初始化（等价于 torch 层的 `grad_x = at::zeros_like(x)`）。

## 函数原型

两段式接口，必须先调用 `aclnnIndexSelectForRank1BackwardGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnIndexSelectForRank1Backward` 接口执行计算。

```c
aclnnStatus aclnnIndexSelectForRank1BackwardGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *index,
    aclTensor *gradX,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnIndexSelectForRank1Backward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnIndexSelectForRank1BackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| gradY（aclTensor） | 输入 | 上游梯度，shape [num_index] | FLOAT32 | ND |
| x（aclTensor） | 输入 | 前向权重表，shape [x_dim0]（仅使用其 shape，数值不参与计算） | FLOAT32 | ND |
| index（aclTensor） | 输入 | 查表索引，shape [num_index]，与 gradY 等长 | INT32 / INT64 | ND |
| gradX（aclTensor） | 输出 | 权重表梯度，shape [x_dim0]，须清零初始化 | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnIndexSelectForRank1Backward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnIndexSelectForRank1BackwardGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- gradY / x / index 必须均为 1D，且 `gradY.shape[0] == index.shape[0]`
- gradY / x / gradX 仅支持 FLOAT32，index 仅支持 INT32 / INT64
- index 元素取值必须在 `[0, x.shape[0])` 内（越界为未定义行为，调用侧保证）
- gradX 必须以全零初始化：kernel 通过 AtomicAdd 向 gradX 累加
- kernel 需将 gradX 全量缓存进 UB，`x.shape[0] * 4B` 加上 20KB 预留须不超过单核 UB 容量
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_index_select_for_rank1_backward.cpp](../examples/aclnn/test_aclnn_index_select_for_rank1_backward.cpp) 的完整实现。
