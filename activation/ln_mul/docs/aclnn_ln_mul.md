# aclnnLnMul

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 训练/推理系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

对输入 X 进行 LayerNorm，再将结果与 gamma、beta 计算，最后与输入 U 做乘法。

计算公式：

```text
output = LayerNorm(X) * U
LayerNorm(X) = rstd * (x - mean) * gamma + beta
```

## 函数原型

两段式接口，必须先调用 `aclnnLnMulGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnLnMul` 接口执行计算。

```c
aclnnStatus aclnnLnMulGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *u,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnLnMul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnLnMulGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| x（aclTensor） | 输入 | 输入数据，shape [A, R] | FLOAT32 | ND |
| u（aclTensor） | 输入 | 输入数据，shape [A, R] | FLOAT32 | ND |
| gamma（aclTensor） | 输入 | LayerNorm 权重，shape [R] | FLOAT32 | ND |
| beta（aclTensor） | 输入 | LayerNorm 偏置，shape [R] | FLOAT32 | ND |
| out（aclTensor） | 输出 | 输出数据，shape [A, R] | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnLnMul

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnLnMulGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 FLOAT32 数据类型
- X、U 必须为 2D 且 shape 相同
- gamma、beta 必须为 1D

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/build_guide.md)。

具体参考 [test_aclnn_ln_mul.cpp](../examples/aclnn/test_aclnn_ln_mul.cpp) 的完整实现。
