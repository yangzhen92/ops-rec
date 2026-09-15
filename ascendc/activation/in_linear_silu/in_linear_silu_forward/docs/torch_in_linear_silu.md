# in_linear_silu

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

## 接口原型

```python
ops_rec.ascendc.activation.in_linear_silu(
    x: Tensor, weight: Tensor, bias: Tensor, split_arg_list: List[int]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

返回 `(user, value, query, key, linear_output)`，支持自动求导（反向内部调用 `aclnnInLinearSiluBackward`）。

另有仅前向的接口，不参与自动求导：

```python
ops_rec.ascendc.activation.distance_in_linear_silu(
    x: Tensor, weight: Tensor, bias: Tensor, split_arg_list: List[int]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| x | 输入 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, k] | 输入数据，k 为 16 的整数倍 |
| weight | 输入 | Tensor | FLOAT16 / BFLOAT16 | ND | [n, k] | 权重，数据类型与 x 保持一致 |
| bias | 输入 | Tensor | FLOAT32 | ND | [n] | 偏置 |
| split_arg_list | 输入 | List[int] | INT64 | - | [H_u, H_v, H_q, H_k] | User/Value/Query/Key 的长度，4 个 int，每个为 16 的整数倍，总和为 n |
| user | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_u] | 输出数据，数据类型与 x 保持一致 |
| value | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_v] | 输出数据，数据类型与 x 保持一致 |
| query | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_q] | 输出数据，数据类型与 x 保持一致 |
| key | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, H_k] | 输出数据，数据类型与 x 保持一致 |
| linear_output | 输出 | Tensor | FLOAT16 / BFLOAT16 | ND | [m, n] | MatMul 中间结果，供反向计算使用 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.activation.in_linear_silu` / `distance_in_linear_silu` 时生效。

### 算子约束

- x、weight 及所有输出仅支持 FLOAT16 / BFLOAT16，bias 仅支持 FLOAT32，weight 数据类型与 x 保持一致
- x、weight 必须为 2D，bias 必须为 1D
- k 取值范围 [16, 8192]，且为 16 的整数倍
- n 取值范围 [64, 32768]，且为 16 的整数倍；n 需大于 k，且为 4\*k 的整数倍
- weight 的第 1 维必须等于 x 的第 1 维（k）
- split_arg_list 长度必须为 4，每个成员取值范围 [16, 8192] 且为 16 的整数倍，4 个成员总和必须等于 n
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [in_linear_silu_plugin.cpp](../../torch_extension/in_linear_silu_plugin.cpp) 在调用算子前完成：

- x、weight 输入 tensor 必须已定义且非空
- x、weight 必须为 2D，bias 若传入必须为 1D
- split_arg_list 长度必须为 4，每个成员取值范围 [16, 8192] 且为 16 的整数倍，总和等于 n
- k 取值范围 [16, 8192] 且为 16 的整数倍，n 取值范围 [64, 32768] 且为 16 的整数倍
- n 必须为 4\*k 的整数倍，weight 的第 1 维必须等于 x 的第 1 维
- bias 若传入，其第 0 维必须等于 weight 的第 0 维（n）
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_in_linear_silu_forward.py](../examples/torch/test_torch_in_linear_silu_forward.py)。
