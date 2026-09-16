# aclnnTokenMixing

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

实现x与其转置x_t相加后的归一化操作。
对输入 `x` 与其转置 `x_t` 求和后做 LayerNorm（带 `gamma` / `beta` / `epsilon`）。

计算公式：

```text
z = x + x_t
y = LayerNorm(z) = rstd * (z - mean) * gamma + beta
```

其中 `x` 为 `[B, S, H]`，`x_t` 为 `[B, H, S]`（通常由调用方对 `x` 做 `permute([0,2,1])` 得到），输出 `y` 为 `[B, H, S]`。

## 函数原型

两段式接口：先调用 `aclnnTokenMixingGetWorkspaceSize` 获取 workspace 与执行器，再调用 `aclnnTokenMixing` 执行计算。

```c
aclnnStatus aclnnTokenMixingGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *xT,
    const aclTensor *gamma,
    const aclTensor *beta,
    double epsilon,
    const aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnTokenMixing(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

> 头文件名以编译 autogen 产物为准（通常为 `aclnn_token_mixing.h`）。属性 `epsilon` 的 C API 形参类型以生成头文件为准。

## aclnnTokenMixingGetWorkspaceSize 接口

### aclnnTokenMixingGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| x（aclTensor） | 输入 | 输入数据，shape [B, S, H] | FLOAT32 | ND |
| xT（aclTensor） | 输入 | `x` 的转置，shape [B, H, S] | FLOAT32 | ND |
| gamma（aclTensor） | 输入 | LayerNorm 权重，shape [H] | FLOAT32 | ND |
| beta（aclTensor） | 输入 | LayerNorm 偏置，shape [H] | FLOAT32 | ND |
| epsilon | 输入 | 防除零常数 | double / float | — |
| y（aclTensor） | 输出 | 输出数据，shape [B, H, S] | FLOAT32 | ND |
| workspaceSize（uint64_t*） | 输出 | Device 侧 workspace 字节数 | — | — |
| executor（aclOpExecutor**） | 输出 | 算子执行器 | — | — |

### aclnnTokenMixingGetWorkspaceSize 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选输入/输出为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 数据类型或格式不在支持范围 |

## aclnnTokenMixing 接口

### aclnnTokenMixing 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | Device 侧 workspace 地址 |
| workspaceSize | 输入 | 由 GetWorkspaceSize 返回的大小 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 执行 Stream |

### aclnnTokenMixing 返回值

`aclnnStatus`：成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 FLOAT32
- `x` 为 3D；`gamma` / `beta` 为 1D 且长度等于 `H`
- 转置混合在 `S == H` 时语义正确
- 需在 NPU 设备上运行

## 调用示例

完整实现见 [test_aclnn_token_mixing.cpp](../examples/aclnn/test_aclnn_token_mixing.cpp)。
