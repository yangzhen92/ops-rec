# aclnnInLinearSilu

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

用于 HSTU Attention 前，将合并并归一化后得到的 UVQK 进行 Linear、Silu 操作后，拆分成 User、Value、Query、Key 4 个 Tensor。

计算公式：

```text
mixed = x @ weight^T + bias
linear_output = mixed
(user, value, query, key) = split(silu(mixed), split_arg_list, dim=-1)
```

## 函数原型

两段式接口，必须先调用 `aclnnInLinearSiluGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnInLinearSilu` 接口执行计算。

```c
aclnnStatus aclnnInLinearSiluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclIntArray *splitArgList,
    const aclTensor *user,
    const aclTensor *value,
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *linearOutput,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnInLinearSilu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnInLinearSiluGetWorkspaceSize 接口

### aclnnInLinearSiluGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -------- | --------- | ------ | --------- | --------- |
| x（aclTensor） | 输入 | 输入数据，shape [m, k] | FLOAT16 / BFLOAT16 | ND |
| weight（aclTensor） | 输入 | 权重，shape [n, k] | FLOAT16 / BFLOAT16 | ND |
| bias（aclTensor） | 输入 | 偏置，shape [n] | FLOAT32 | ND |
| splitArgList（aclIntArray*） | 输入 | 拆分长度，4 个 int：[H_u, H_v, H_q, H_k]，总和为 n | INT64 | - |
| user（aclTensor） | 输出 | 输出数据，shape [m, H_u] | FLOAT16 / BFLOAT16 | ND |
| value（aclTensor） | 输出 | 输出数据，shape [m, H_v] | FLOAT16 / BFLOAT16 | ND |
| query（aclTensor） | 输出 | 输出数据，shape [m, H_q] | FLOAT16 / BFLOAT16 | ND |
| key（aclTensor） | 输出 | 输出数据，shape [m, H_k] | FLOAT16 / BFLOAT16 | ND |
| linearOutput（aclTensor） | 输出 | MatMul 中间结果，shape [m, n] | FLOAT16 / BFLOAT16 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### aclnnInLinearSiluGetWorkspaceSize 返回值

| 返回值 | 错误码 | 描述 |
| -------- | ------- | ------ |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnInLinearSilu 接口

### aclnnInLinearSilu 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | ------ |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnInLinearSiluGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### aclnnInLinearSilu 返回值

`aclnnStatus`：返回状态码，成功返回 `ACLNN_SUCCESS`。

## aclnnInLinearSilu 约束说明

- x、weight 及所有输出仅支持 FLOAT16 / BFLOAT16，bias 仅支持 FLOAT32，weight 数据类型与 x 保持一致
- x、weight 必须为 2D，bias 必须为 1D
- k 取值范围 [16, 8192]，且为 16 的整数倍
- n 取值范围 [64, 32768]，且为 16 的整数倍；n 需大于 k，且为 4\*k 的整数倍
- weight 的第 1 维必须等于 x 的第 1 维（k）
- splitArgList 长度必须为 4，每个成员取值范围 [16, 8192] 且为 16 的整数倍，4 个成员总和必须等于 n
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_in_linear_silu_forward.cpp](../examples/aclnn/test_aclnn_in_linear_silu_forward.cpp) 的完整实现。
