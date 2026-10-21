# aclnnNormMultiplyDropout

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

对输入 x 做 LayerNorm（含 weight/bias 仿射变换），结果与 u 相乘后按 dropout_ratio 做 dropout。

计算公式：

```text
ln_ret = LayerNorm(x) * weight + bias
ln_out = u * ln_ret
output = dropout(ln_out, p=dropout_ratio)
```

## 函数原型

两段式接口，必须先调用 `aclnnNormMultiplyDropoutGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnNormMultiplyDropout` 接口执行计算。

```c
aclnnStatus aclnnNormMultiplyDropoutGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *u,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *mask,
    float eps,
    float dropoutRatio,
    const aclTensor *output,
    const aclTensor *mean,
    const aclTensor *var,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnNormMultiplyDropout(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnNormMultiplyDropoutGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| x（aclTensor） | 输入 | 输入数据，shape [B, C] | FLOAT32/FLOAT16/BFLOAT16（Ascend 950PR&950DT系列产品不支持 FLOAT32） | ND |
| u（aclTensor） | 输入 | 乘法因子，shape [B, C]，数据类型与 x 一致 | 同 x | ND |
| weight（aclTensor） | 输入 | LayerNorm 权重，shape [C]，数据类型与 x 一致 | 同 x | ND |
| bias（aclTensor） | 输入 | LayerNorm 偏置，shape [C]，数据类型与 x 一致 | 同 x | ND |
| mask（aclTensor） | 输入 | dropout 掩码，约为 x 元素个数的 1/8；dropout_ratio <= 1e-10 时可为空 tensor | UINT8 | ND |
| eps（float） | 输入 | 防除 0 极小值，取值范围 [1e-10, 1e-4] | - | - |
| dropoutRatio（float） | 输入 | dropout 概率，取值范围 [0.0, 1.0]；<= 1e-10 时不 dropout | - | - |
| output（aclTensor） | 输出 | 前向计算结果，shape [B, C]，数据类型与 x 一致 | 同 x | ND |
| mean（aclTensor） | 输出 | 归一化均值，shape [B] | FLOAT32 | ND |
| var（aclTensor） | 输出 | 归一化方差，shape [B] | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnNormMultiplyDropout

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnNormMultiplyDropoutGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 输入 tensor 值域 [-1, 1)，且数据必须连续（contiguous）
- B 取值范围 [1, 1000000]，C 仅支持 512、1024
- x/u/weight/bias 数据类型必须一致
- **Ascend 950PR&950DT系列产品不支持输入 x 的 FLOAT32 数据类型**
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 torch 适配层 [norm_multiply_dropout_plugin.cpp](../../torch_extension/norm_multiply_dropout_plugin.cpp) 中 `EXEC_NPU_CMD` 的两段式调用实现。
