# torch_norm_multiply_dropout

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

norm_multiply_dropout 的 torch 接口。实现 LayerNorm + multiply + dropout 计算逻辑对应的融合算子功能，支持自动求导与 `torch.compile`（inductor）图模式调用。

计算公式：

```text
ln_ret = LayerNorm(x) * weight + bias
ln_out = u * ln_ret
output = dropout(ln_out, p=dropout_ratio)
```

## 函数原型

```python
ops_rec.ascendc.activation.norm_multiply_dropout(
    x,
    u,
    weight,
    bias,
    eps=1e-05,
    dropout_ratio=0.0
) -> (output, mean, var)
```

## 参数说明

| 参数名 | 输入/输出 | 类型 | 数据类型 | shape | 说明 |
|--------|---------|------|---------|-------|------|
| x | 输入 | Tensor | float32/float16/bfloat16 | [B, C] | 输入数据 |
| u | 输入 | Tensor | 同 x | [B, C] | 乘法因子 |
| weight | 输入 | Tensor | 同 x | [C] | LayerNorm 权重 |
| bias | 输入 | Tensor | 同 x | [C] | LayerNorm 偏置 |
| eps | 输入 | float | - | - | 防除 0 极小值，默认 1e-5，取值范围 [1e-10, 1e-4] |
| dropout_ratio | 输入 | float | - | - | dropout 概率，默认 0.0，取值范围 [0.0, 1.0] |
| output | 输出 | Tensor | 同 x | [B, C] | 前向计算结果 |
| mean | 输出 | Tensor | float32 | [B] | 归一化均值 |
| var | 输出 | Tensor | float32 | [B] | 归一化方差 |

## 使用说明

- dropout 掩码 mask 由接口自动生成，无需用户手动构造
- 支持自动求导：反向通过 `ops_rec::norm_multiply_dropout_backward` 算子完成
- 支持 `torch.compile(model, fullgraph=True)`（inductor 图模式），已注册 Meta 实现
- Ascend 950PR&950DT系列产品不支持输入 x 的 float32 数据类型

## 约束说明

- 输入 tensor 值域 [-1, 1)，且数据必须连续（contiguous）
- B 取值范围 [1, 1000000]，C 仅支持 512、1024
- 该算子需要在 NPU 设备上运行

## 调用示例

参考 [test_torch_norm_multiply_dropout.py](../examples/torch/test_torch_norm_multiply_dropout.py)。
