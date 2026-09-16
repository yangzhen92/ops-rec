# in_linear_silu_backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3 系列产品 | √ |
| Atlas A2 系列产品 | √ |

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

## 接口原型

```python
ops_rec.ascendc.activation.in_linear_silu_backward(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    user_grad: Tensor,
    value_grad: Tensor,
    query_grad: Tensor,
    key_grad: Tensor,
    linear_output: Tensor,
    attr_dict: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

返回值为 `(x_grad, weights_grad, bias_grad, output)` 四元组：`x_grad` 数据类型与 x 一致，`weights_grad` 数据类型与 weight 一致，`bias_grad` 为 FLOAT32（未传入 bias 时为未定义 tensor），第四个元素为占位用的未定义 tensor。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| x | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, embed_dim] | 输入数据 |
| weight | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [hidden_size, embed_dim] | 权重，数据类型与 x 一致 |
| bias | 输入 | Tensor | FLOAT32 | ND | [hidden_size] | 偏置，可选参数 |
| user_grad | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, H_u] | user 分量上游梯度，数据类型与 x 一致 |
| value_grad | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, H_v] | value 分量上游梯度，数据类型与 x 一致 |
| query_grad | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, H_q] | query 分量上游梯度，数据类型与 x 一致 |
| key_grad | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, H_k] | key 分量上游梯度，数据类型与 x 一致 |
| linear_output | 输入 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, hidden_size] | 前向传播线性层输出，数据类型与 x 一致 |
| attr_dict | 输入 | List[int] | INT64 | - | [H_u, H_v, H_q, H_k] | split 维度列表，长度为 4 |
| x_grad | 输出 | Tensor | FLOAT16/BFLOAT16 | ND | [seq_len, embed_dim] | x 的梯度，数据类型与 x 一致 |
| weights_grad | 输出 | Tensor | FLOAT16/BFLOAT16 | ND | [hidden_size, embed_dim] | weight 的梯度，数据类型与 weight 一致 |
| bias_grad | 输出 | Tensor | FLOAT32 | ND | [hidden_size] | bias 的梯度，仅当输入包含 bias 时有效 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.activation.in_linear_silu_backward` 时生效。

### 算子约束

- x、weight、user_grad、value_grad、query_grad、key_grad、linear_output 仅支持 FLOAT16/BFLOAT16 数据类型，且数据类型一致；bias 仅支持 FLOAT32
- x 必须为 2D，shape 为 [seq_len, embed_dim]，embed_dim 取值范围 [16, 8192] 且为 16 的整数倍
- weight 必须为 2D，shape 为 [hidden_size, embed_dim]，hidden_size 取值范围 [64, 32768] 且为 16 的整数倍，且等于 sum(attr_dict)
- attr_dict 长度必须为 4，每个元素取值范围 [16, 8192] 且为 16 的整数倍
- user_grad、value_grad、query_grad、key_grad、linear_output 必须为 2D，dim[0] 与 x 的 dim[0] 一致
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [in_linear_silu_backward_plugin.cpp](../../torch_extension/in_linear_silu_backward_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- user_grad、value_grad、query_grad、key_grad 的 dim[1] 必须与 attr_dict 中对应元素相等
- linear_output 的 dim[1] 必须等于 weight 的 dim[0]
- weight 的 dim[0] 必须是 4 倍 x 的 dim[1] 的整数倍
- 变维场景（attr_dict 四个元素不全相等）由扩展层自动完成梯度拼接，对调用方透明
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_in_linear_silu_backward.py](../examples/torch/test_torch_in_linear_silu_backward.py)。
