# aclnnHstuDenseBackward (Normal Layout).txt

# aclnnHstuDenseBackward (Normal Layout)

## 产品支持情况

| 产品              | 是否支持 |
|:----------------| :------: |
| Ascend 950PR&950DT 系列产品  | √ |
| Atlas A3 训练系列产品 | √ |
| Atlas A2 训练系列产品 | √ |

## 功能说明

HSTU (Hierarchical State Transition Unit) Dense Backward 算子，Normal Layout 反向传播。

输入 shape [B, S, N, D]，适用于固定长度序列。

## 函数原型

两段式接口，必须先调用 `aclnnHstuDenseBackwardGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnHstuDenseBackward` 执行计算。

```c
aclnnStatus aclnnHstuDenseBackwardGetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *mask,
    const aclTensor *attnBias,
    const char *layout,
    const int maskType,
    const int maxSeqLen,
    const float siluScale,
    const int *targetGroupSize,
    const float *alpha,
    aclTensor *qGrad,
    aclTensor *kGrad,
    aclTensor *vGrad,
    aclTensor *attnBiasGrad,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnHstuDenseBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnHstuDenseBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| grad | 输入 | 上游梯度，shape [B, S, N, D] | FLOAT32/FLOAT16/BF16 | ND |
| q | 输入 | Query tensor，shape [B, S, N, D] | 同 grad | ND |
| k | 输入 | Key tensor，shape [B, S, N, D] | 同 grad | ND |
| v | 输入 | Value tensor，shape [B, S, N, D] | 同 grad | ND |
| mask | 输入 | 注意力掩码（可选），shape [B, N, S, S] | 同 grad | ND |
| attnBias | 输入 | 注意力偏置（可选），shape [B, N, S, S] | 同 grad | ND |
| layout | 输入 | 布局类型，固定为 "normal" | string | - |
| maskType | 输入 | 掩码类型：0=tril, 2=none, 3=custom | int | - |
| maxSeqLen | 输入 | 最大序列长度 | int | - |
| siluScale | 输入 | SiLU 激活缩放因子 | float | - |
| targetGroupSize | 输入 | 目标分组大小（可选） | int* | - |
| alpha | 输入 | 缩放因子 alpha（可选） | float* | - |
| qGrad | 输出 | Q 的梯度，shape [B, S, N, D] | 同 grad | ND |
| kGrad | 输出 | K 的梯度，shape [B, S, N, D] | 同 grad | ND |
| vGrad | 输出 | V 的梯度，shape [B, S, N, D] | 同 grad | ND |
| attnBiasGrad | 输出 | 注意力偏置的梯度（可选） | 同 grad | ND |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | uint64_t* | - |
| executor | 输出 | 返回算子执行器 | aclOpExecutor** | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnHstuDenseBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnHstuDenseBackwardGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 支持 FLOAT32、FLOAT16、BF16 数据类型
- 输入必须为 4D [B, S, N, D]
- mask_type 支持：0 (tril), 2 (none), 3 (custom)
- head_dim 必须为 16 的倍数
- 该算子需要在 NPU 设备上运行

## 调用示例

参考 [test_aclnn_hstu_dense_backward.cpp](../examples/aclnn/test_aclnn_hstu_backward_v1_dense.cpp)
