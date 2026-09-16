# aclnnHstuJaggedBackward (Jagged Layout).txt

# aclnnHstuJaggedBackward (Jagged Layout)

## 产品支持情况

| 产品              | 是否支持 |
|:----------------| :------: |
| Ascend 950PR&950DT系列产品  | √ |
| Atlas A3训练系列产品 | √ |
| Atlas A2训练系列产品 | √ |

## 功能说明

HSTU (Hierarchical State Transition Unit) Jagged Backward 算子，Jagged Layout 反向传播。

输入 shape [T, N, D]，适用于变长序列，其中 T 为所有 batch 的序列长度之和。

## 函数原型

两段式接口，必须先调用 `aclnnHstuJaggedBackwardGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnHstuJaggedBackward` 执行计算。

```c
aclnnStatus aclnnHstuJaggedBackwardGetWorkspaceSize(
    const aclTensor *grad,
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *mask,
    const aclTensor *attnBias,
    const aclTensor *seqOffsetQ,
    const aclTensor *seqOffsetK,
    const aclTensor *numContext,
    const aclTensor *numTarget,
    const char *layout,
    const int maskType,
    const int maxSeqLenQ,
    const int maxSeqLenK,
    const float siluScale,
    const int *targetGroupSize,
    const float *alpha,
    aclTensor *qGrad,
    aclTensor *kGrad,
    aclTensor *vGrad,
    aclTensor *attnBiasGrad,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnHstuJaggedBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnHstuJaggedBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| grad | 输入 | 上游梯度，shape [T, N, D_v] | FLOAT32/FLOAT16/BF16 | ND |
| q | 输入 | Query tensor，shape [T, N, D_qk] | 同 grad | ND |
| k | 输入 | Key tensor，shape [T, N, D_qk] | 同 grad | ND |
| v | 输入 | Value tensor，shape [T, N, D_v] | 同 grad | ND |
| mask | 输入 | 注意力掩码（可选），shape [B, N, max_seq_q, max_seq_k] | 同 grad | ND |
| attnBias | 输入 | 注意力偏置（可选），shape [B, N, max_seq_q, max_seq_k] | 同 grad | ND |
| seqOffsetQ | 输入 | Q 的序列偏移，shape [B+1] | int32/int64 | ND |
| seqOffsetK | 输入 | K 的序列偏移，shape [B+1] | int32/int64 | ND |
| numContext | 输入 | Context 掩码长度（可选） | int32/int64 | ND |
| numTarget | 输入 | Target 掩码长度（可选） | int32/int64 | ND |
| layout | 输入 | 布局类型，固定为 "jagged" | string | - |
| maskType | 输入 | 掩码类型：0=tril, 2=none, 3=custom | int | - |
| maxSeqLenQ | 输入 | Q 的最大序列长度 | int | - |
| maxSeqLenK | 输入 | K 的最大序列长度 | int | - |
| siluScale | 输入 | SiLU 激活缩放因子 | float | - |
| targetGroupSize | 输入 | 目标分组大小（可选） | int* | - |
| alpha | 输入 | 缩放因子 alpha（可选） | float* | - |
| qGrad | 输出 | Q 的梯度，shape [T, N, D_qk] | 同 grad | ND |
| kGrad | 输出 | K 的梯度，shape [T, N, D_qk] | 同 grad | ND |
| vGrad | 输出 | V 的梯度，shape [T, N, D_v] | 同 grad | ND |
| attnBiasGrad | 输出 | 注意力偏置的梯度（可选） | 同 grad | ND |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | uint64_t* | - |
| executor | 输出 | 返回算子执行器 | aclOpExecutor** | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnHstuJaggedBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnHstuJaggedBackwardGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 支持 FLOAT32、FLOAT16、BF16 数据类型
- 输入必须为 3D [T, N, D]
- head_dim 必须为 16 的倍数
- mask_type 支持：0 (tril), 2 (none), 3 (custom)
- seq_offset_q 和 seq_offset_k 必须提供，长度为 [B+1]
- 该算子需要在 NPU 设备上运行

## 调用示例

参考 [test_aclnn_hstu_jagged_backward.cpp](../examples/aclnn/test_aclnn_hstu_backward_v1_jagged.cpp)
