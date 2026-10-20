# aclnnNormMultiplyDropoutBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

计算 LayerNorm + multiply + dropout 前向融合算子的反向梯度，输出 d_u、d_x、d_weight、d_bias。

## 函数原型

两段式接口，必须先调用 `aclnnNormMultiplyDropoutBackwardGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnNormMultiplyDropoutBackward` 接口执行计算。

```c
aclnnStatus aclnnNormMultiplyDropoutBackwardGetWorkspaceSize(
    const aclTensor *dOut,
    const aclTensor *x,
    const aclTensor *u,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *mean,
    const aclTensor *var,
    const aclTensor *mask,
    float eps,
    float dropoutRatio,
    const aclTensor *dU,
    const aclTensor *dX,
    const aclTensor *dWeight,
    const aclTensor *dBias,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnNormMultiplyDropoutBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnNormMultiplyDropoutBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| dOut（aclTensor） | 输入 | 上游梯度，shape [B, C]，数据类型与 x 一致 | FLOAT32/FLOAT16/BFLOAT16 | ND |
| x（aclTensor） | 输入 | 前向输入，shape [B, C] | 同 dOut | ND |
| u（aclTensor） | 输入 | 前向乘法因子，shape [B, C]，数据类型与 x 一致 | 同 dOut | ND |
| weight（aclTensor） | 输入 | 前向 LayerNorm 权重，shape [C] | 同 dOut | ND |
| bias（aclTensor） | 输入 | 前向 LayerNorm 偏置，shape [C] | 同 dOut | ND |
| mean（aclTensor） | 输入 | 前向输出的均值，shape [B] | FLOAT32 | ND |
| var（aclTensor） | 输入 | 前向输出的方差，shape [B] | FLOAT32 | ND |
| mask（aclTensor） | 输入 | 前向 dropout 掩码 | UINT8 | ND |
| eps（float） | 输入 | 防除 0 极小值，取值范围 [1e-10, 1e-4] | - | - |
| dropoutRatio（float） | 输入 | dropout 概率，取值范围 [0.0, 1.0] | - | - |
| dU（aclTensor） | 输出 | u 的梯度，shape [B, C]，数据类型与 x 一致 | 同 dOut | ND |
| dX（aclTensor） | 输出 | x 的梯度，shape [B, C]，数据类型与 x 一致 | 同 dOut | ND |
| dWeight（aclTensor） | 输出 | weight 的梯度，shape [C] | FLOAT32 | ND |
| dBias（aclTensor） | 输出 | bias 的梯度，shape [C] | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnNormMultiplyDropoutBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnNormMultiplyDropoutBackwardGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 除 dOut 外，其他输入均为前向计算的入参/输出，由 PyTorch 自动求导框架保存并传入
- B 取值范围 [1, 1000000]，C 仅支持 512、1024
- **Ascend 950PR&950DT系列产品不支持输入 x 的 FLOAT32 数据类型**
- 该算子不建议单独调用，建议通过 PyTorch 自动求导框架调用

## 调用示例

具体参考 torch 适配层 [norm_multiply_dropout_backward_plugin.cpp](../torch_extension/norm_multiply_dropout_backward_plugin.cpp) 中 `EXEC_NPU_CMD` 的两段式调用实现。
