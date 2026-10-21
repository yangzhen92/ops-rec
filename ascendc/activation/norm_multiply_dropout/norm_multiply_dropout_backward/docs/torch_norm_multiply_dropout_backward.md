# torch_norm_multiply_dropout_backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

norm_multiply_dropout 前向算子的反向求导 torch 接口。一般不建议单独调用，推荐通过前向接口的自动求导机制（`loss.backward()`）触发反向算子执行。

## 函数原型

```python
torch.ops.ops_rec.norm_multiply_dropout_backward(
    d_out,
    x,
    u,
    weight,
    bias,
    mean,
    var,
    mask,
    eps=1e-05,
    dropout_ratio=0.0
) -> (d_u, d_x, d_weight, d_bias)
```

## 参数说明

| 参数名 | 输入/输出 | 类型 | 数据类型 | shape | 说明 |
|--------|---------|------|---------|-------|------|
| d_out | 输入 | Tensor | 同 x | [B, C] | 上游梯度 |
| x | 输入 | Tensor | float32/float16/bfloat16 | [B, C] | 前向输入 |
| u | 输入 | Tensor | 同 x | [B, C] | 前向乘法因子 |
| weight | 输入 | Tensor | 同 x | [C] | 前向 LayerNorm 权重 |
| bias | 输入 | Tensor | 同 x | [C] | 前向 LayerNorm 偏置 |
| mean | 输入 | Tensor | float32 | [B] | 前向输出的均值 |
| var | 输入 | Tensor | float32 | [B] | 前向输出的方差 |
| mask | 输入 | Tensor | uint8 | [B, C/8] | 前向 dropout 掩码 |
| eps | 输入 | float | - | - | 防除 0 极小值，默认 1e-5 |
| dropout_ratio | 输入 | float | - | - | dropout 概率，默认 0.0 |
| d_u | 输出 | Tensor | 同 x | [B, C] | u 的梯度 |
| d_x | 输出 | Tensor | 同 x | [B, C] | x 的梯度 |
| d_weight | 输出 | Tensor | float32 | [C] | weight 的梯度，为保持累加精度固定为 float32 |
| d_bias | 输出 | Tensor | float32 | [C] | bias 的梯度，为保持累加精度固定为 float32 |

## 使用说明

- 除 d_out 外，其他输入均为前向计算的入参/输出，由 PyTorch 自动求导框架保存并传入
- 支持 `torch.compile(model, fullgraph=True)`（inductor 图模式），已注册 Meta 实现
- Ascend 950PR&950DT系列产品不支持输入 x 的 float32 数据类型

## 约束说明

- 输入 tensor 值域 [-1, 1)，且数据必须连续（contiguous）
- B 取值范围 [1, 1000000]，C 仅支持 512、1024
- 该算子需要在 NPU 设备上运行

## 调用示例

参考 [test_torch_norm_multiply_dropout_backward.py](../test/pytest/test_norm_multiply_dropout_backward.py)。
