# in_linear_silu_forward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3 训练/推理系列产品 | √ |
| Atlas A2 训练/推理系列产品 | √ |

## 功能说明

用于 HSTU Attention 前，将合并并归一化后得到的 UVQK 进行 Linear、Silu 操作后，拆分成 User、Value、Query、Key 4 个 Tensor。

计算公式：

```text
mixed = x @ weight^T + bias
linear_output = mixed
(user, value, query, key) = split(silu(mixed), split_arg_list, dim=-1)
```

对应接口：

- aclnn 接口：`aclnnInLinearSilu`（反向为 `aclnnInLinearSiluBackward`）
- torch 接口：`ops_rec.ascendc.activation.in_linear_silu`（支持自动求导）
- torch 接口（仅前向）：`ops_rec.ascendc.activation.distance_in_linear_silu`

## 算子实现原理

算子基于 CATLASS 与 AscendC 开发，将 MatMul、Silu、split 融合在单个 Kernel 内完成。具体计算步骤如下：

1. 输入张量 x（shape [m, k]）、weight（shape [n, k]）、bias（shape [n]）
2. 计算 MatMul 结果 `mixed = x @ weight^T + bias`，shape [m, n]
3. 输出中间结果 `linear_output = mixed`，用于反向计算或 workspace
4. 对 mixed 计算 Silu 激活
5. 根据 split_arg_list（[H_u, H_v, H_q, H_k]，sum = n）沿最后一维将 Silu 结果拆分为 user、value、query、key 四个输出

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| x | 输入 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, k] | 输入数据，k 为 16 的整数倍 |
| weight | 输入 | Tensor | FLOAT16 / BFLOAT16 | ND | [n, k] | 权重，数据类型与 x 保持一致；若为 [k, n] 需转置后传入 |
| bias | 输入 | Tensor | FLOAT32 | ND | [n] | 偏置 |
| split_arg_list | 输入（属性） | ListInt | INT64 | - | [H_u, H_v, H_q, H_k] | User/Value/Query/Key 的长度，4 个 int，每个为 16 的整数倍，总和为 n |
| user | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_u] | 输出数据，数据类型与 x 保持一致 |
| value | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_v] | 输出数据，数据类型与 x 保持一致 |
| query | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_q] | 输出数据，数据类型与 x 保持一致 |
| key | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_k] | 输出数据，数据类型与 x 保持一致 |
| linear_output | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, n] | MatMul 中间结果，用于反向计算或 workspace |

## 约束说明

- x、weight 及所有输出仅支持 FLOAT16 / BFLOAT16，bias 仅支持 FLOAT32，weight 数据类型与 x 保持一致
- x、weight 必须为 2D，bias 必须为 1D
- k 取值范围 [16, 8192]，且为 16 的整数倍
- n 取值范围 [64, 32768]，且为 16 的整数倍；n 需大于 k，且为 4\*k 的整数倍
- weight 的第 1 维必须等于 x 的第 1 维（k）
- split_arg_list 长度必须为 4，每个成员取值范围 [16, 8192] 且为 16 的整数倍，4 个成员总和必须等于 n
- 算子输入 shape 受显存大小限制
- 该算子需要在 NPU 设备上运行
- 编译依赖 CATLASS 源码，编译前需初始化 submodule：

```bash
git submodule update --init --recursive
```

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_in_linear_silu.cpp)
- [torch 调用示例](./examples/torch/test_torch_in_linear_silu.py)
- [torch 功能测试](./test/test_in_linear_silu.py)
- [aclnn 接口文档](./docs/aclnn_in_linear_silu.md)
- [torch 接口文档](./docs/torch_in_linear_silu.md)
