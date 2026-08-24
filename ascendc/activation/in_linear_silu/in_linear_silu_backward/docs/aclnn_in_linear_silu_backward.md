# aclnnInLinearSiluBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

in_linear_silu_backward 是 in_linear_silu 算子的反向传播实现，用于计算输入 x、权重 weight 和偏置 bias 的梯度。

计算公式：

```text
mixed = concat(user_grad, value_grad, query_grad, key_grad)
lo_grad = mixed * silu_grad(linear_output)
x_grad = lo_grad @ weight
weights_grad = lo_grad^T @ x
bias_grad = reduce_sum(lo_grad, dim=0)
silu_grad(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

## 函数原型

两段式接口，必须先调用 `aclnnInLinearSiluBackwardGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnInLinearSiluBackward` 接口执行计算。

```c
aclnnStatus aclnnInLinearSiluBackwardGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *userGrad,
    const aclTensor *valueGrad,
    const aclTensor *queryGrad,
    const aclTensor *keyGrad,
    const aclTensor *linearOutput,
    const aclIntArray *splitArgList,
    const bool *isTrans,
    const bool *isVardim,
    const aclTensor *xGrad,
    const aclTensor *weightGrad,
    const aclTensor *biasGrad,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnInLinearSiluBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnInLinearSiluBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| x（aclTensor） | 输入 | 输入数据，shape [seq_len, embed_dim] | FLOAT16/BFLOAT16 | ND |
| weight（aclTensor） | 输入 | 权重，shape [hidden_size, embed_dim]，数据类型与 x 一致 | FLOAT16/BFLOAT16 | ND |
| bias（aclTensor） | 输入 | 偏置，shape [hidden_size]，可选参数，不使用时传空 | FLOAT32 | ND |
| userGrad（aclTensor） | 输入 | user 分量上游梯度，shape [seq_len, H_u] | FLOAT16/BFLOAT16 | ND |
| valueGrad（aclTensor） | 输入 | value 分量上游梯度，shape [seq_len, H_v] | FLOAT16/BFLOAT16 | ND |
| queryGrad（aclTensor） | 输入 | query 分量上游梯度，shape [seq_len, H_q] | FLOAT16/BFLOAT16 | ND |
| keyGrad（aclTensor） | 输入 | key 分量上游梯度，shape [seq_len, H_k] | FLOAT16/BFLOAT16 | ND |
| linearOutput（aclTensor） | 输入 | 前向传播线性层输出，shape [seq_len, hidden_size] | FLOAT16/BFLOAT16 | ND |
| splitArgList（aclIntArray*） | 输入 | split 维度列表 [H_u, H_v, H_q, H_k]，长度为 4 | INT64 | - |
| isTrans（bool*） | 输入 | 权重是否需要转置，保留参数 | BOOL | - |
| isVardim（bool*） | 输入 | 是否变维场景，为 true 时表示四个梯度已按 [H_u, H_v, H_q, H_k] 拼接为一个 tensor 由 userGrad 传入 | BOOL | - |
| xGrad（aclTensor） | 输出 | x 的梯度，shape [seq_len, embed_dim] | FLOAT32 | ND |
| weightGrad（aclTensor） | 输出 | weight 的梯度，shape [hidden_size, embed_dim] | FLOAT32 | ND |
| biasGrad（aclTensor） | 输出 | bias 的梯度，shape [hidden_size]，仅输入包含 bias 时有效 | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnInLinearSiluBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnInLinearSiluBackwardGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- x、weight、userGrad、valueGrad、queryGrad、keyGrad、linearOutput 仅支持 FLOAT16/BFLOAT16 数据类型，且数据类型一致；bias 仅支持 FLOAT32
- x 必须为 2D，shape 为 [seq_len, embed_dim]，embed_dim 取值范围 [16, 8192] 且为 16 的整数倍
- weight 必须为 2D，shape 为 [hidden_size, embed_dim]，hidden_size 取值范围 [64, 32768] 且为 16 的整数倍，且等于 sum(splitArgList)
- splitArgList 长度必须为 4，每个元素取值范围 [16, 8192] 且为 16 的整数倍
- userGrad、valueGrad、queryGrad、keyGrad、linearOutput 必须为 2D，dim[0] 与 x 的 dim[0] 一致
- 输出 xGrad、weightGrad、biasGrad 均为 FLOAT32
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_in_linear_silu_backward.cpp](../examples/aclnn/test_aclnn_in_linear_silu_backward.cpp) 的完整实现。
