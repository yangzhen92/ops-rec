# hstu_dense_backward (Normal Layout)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3 训练/推理系列产品 | √ |
| Atlas A2 训练/推理系列产品 | √ |

## 功能说明

HSTU (Hierarchical State Transition Unit) Dense Backward 算子，Normal Layout 反向传播。

输入 shape [B, S, N, D]，适用于固定长度序列。

## 接口原型

```python
ops_rec.ascendc.attention.hstu_dense_backward(
    grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    attn_bias: Optional[Tensor] = None,
    mask_type: int = 0,
    max_seq_len: int = 0,
    silu_scale: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| grad | 输入 | Tensor | FLOAT32/FLOAT16/BF16 | ND | [B, S, N, D] | 上游梯度 |
| q | 输入 | Tensor | 同 grad | ND | [B, S, N, D] | Query tensor |
| k | 输入 | Tensor | 同 grad | ND | [B, S, N, D] | Key tensor |
| v | 输入 | Tensor | 同 grad | ND | [B, S, N, D] | Value tensor |
| mask | 输入 | Optional[Tensor] | 同 grad | ND | [B, N, S, S] | 注意力掩码（可选） |
| attn_bias | 输入 | Optional[Tensor] | 同 grad | ND | [B, N, S, S] | 注意力偏置（可选） |
| mask_type | 输入 | int | - | - | - | 掩码类型：0=tril, 2=none, 3=custom |
| max_seq_len | 输入 | int | - | - | - | 最大序列长度 |
| silu_scale | 输入 | float | - | - | - | SiLU 激活缩放因子 |
| q_grad | 输出 | Tensor | 同 grad | ND | [B, S, N, D] | Q 的梯度 |
| k_grad | 输出 | Tensor | 同 grad | ND | [B, S, N, D] | K 的梯度 |
| v_grad | 输出 | Tensor | 同 grad | ND | [B, S, N, D] | V 的梯度 |
| attn_bias_grad | 输出 | Optional[Tensor] | 同 grad | ND | [B, N, S, S] | 注意力偏置的梯度 |

## 约束说明

### 算子约束

- 支持 FLOAT32、FLOAT16、BF16 数据类型
- 输入必须为 4D [B, S, N, D]
- mask_type 支持：0 (tril), 2 (none), 3 (custom)
- head_dim 必须为 16 的倍数
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 torch_extension 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- mask_type 必须在 [0, 3] 范围内
- max_seq_len 必须在 [1, 20480] 范围内
- 需要安装 `torch_npu`

## 调用示例

参考 [test_torch_hstu_dense_backward.py](../examples/torch/test_torch_hstu_backward_v1_dense.py)
