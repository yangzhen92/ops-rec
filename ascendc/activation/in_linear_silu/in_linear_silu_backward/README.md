# in_linear_silu_backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3 系列产品 | √ |
| Atlas A2 系列产品 | √ |

## 功能说明

in_linear_silu_backward是in_linear_silu算子的反向传播实现，用于计算输入x、权重weight和偏置bias的梯度。

## 接口说明

- aclnn 接口：`aclnnInLinearSiluBackward`
- torch 接口：`ops_rec.ascendc.activation.in_linear_silu_backward`

## 算子输入与输出

| 名称      | 输入/输出 | 参数类型 | 数据类型         | 数据格式       | 范围         | 说明                                  |
|---------|------------|------|--------------|------------|------------|----------------------------------------|
| x       | 输入       | Tensor | float16/bfloat16 | [seq_len, embed_dim] | embed_dim取值范围[16, 8192] |   embed_dim为16的整数倍              |
| weight   | 输入       | Tensor | float16/bfloat16 | [hidden_size, embed_dim] | hidden_size取值范围[64, 32768] |   hidden_size为16的整数倍，hidden_size = sum(split_arg_list), 数据类型与x保持一致            |
| bias    | 输入       | Tensor | float32 | [hidden_size] | hidden_size取值范围[64, 32768] |   hidden_size为16的整数倍，可选参数            |
| user_grad | 输入       | Tensor | float16/bfloat16 | [seq_len, H_u] | H_u取值范围[16, 8192] |   H_u为16的整数倍, 数据类型与x保持一致          |
| value_grad | 输入       | Tensor | float16/bfloat16 | [seq_len, H_v] | H_v取值范围[16, 8192] |   H_v为16的整数倍, 数据类型与x保持一致          |
| query_grad | 输入       | Tensor | float16/bfloat16 | [seq_len, H_q] | H_q取值范围[16, 8192] |   H_q为16的整数倍 , 数据类型与x保持一致         |
| key_grad | 输入       | Tensor | float16/bfloat16 | [seq_len, H_k] | H_k取值范围[16, 8192] |   H_k为16的整数倍 , 数据类型与x保持一致         |
| linear_output | 输入       | Tensor | float16/bfloat16 | [seq_len, hidden_size] | hidden_size取值范围[64, 32768] |   hidden_size为16的整数倍，来自前向传播的中间结果，数据类型与x保持一致            |
| split_arg_list | 输入(属性)  | List[int] | int   | [H_u, H_v, H_q, H_k]          |   sum(split_arg_list)=hidden_size       |长度为4，不可为空，每个元素为16的整数倍     |
| isTrans | 输入(属性)  | Bool | bool   | -          |   保留参数，当前仅支持为true       |权重是否需要转置     |
| x_grad | 输出     | Tensor | float16/bfloat16 | [seq_len, embed_dim] | embed_dim取值范围[16, 8192] |   embed_dim为16的整数倍, 数据类型与x保持一致          |
| weights_grad | 输出     | Tensor | float16/bfloat16 | [hidden_size, embed_dim] | hidden_size取值范围[64, 32768] |   hidden_size为16的整数倍, 数据类型与weight保持一致          |
| bias_grad | 输出     | Tensor | float32 | [hidden_size] | hidden_size取值范围[64, 32768] |   hidden_size为16的整数倍, 仅当输入包含bias时输出         |

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_in_linear_silu_backward.cpp)
- [torch 调用示例](./examples/torch/test_torch_in_linear_silu_backward.py)
- [aclnn 接口文档](./docs/aclnn_in_linear_silu_backward.md)
- [torch 接口文档](./docs/torch_in_linear_silu_backward.md)
