# hstu_jagged_backward (Jagged Layout)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3 训练/推理系列产品 | √ |
| Atlas A2 训练/推理系列产品 | √ |

## 功能说明

HSTU (Hierarchical State Transition Unit) Jagged Backward 算子，Jagged Layout 反向传播。

输入 shape [T, N, D]，适用于变长序列，其中 T 为所有 batch 的序列长度之和。

## 接口原型

```python
ops_rec.ascendc.attention.hstu_jagged_backward(
    grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    attn_bias: Optional[Tensor] = None,
    mask_type: int = 0,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
    silu_scale: float = 0.0,
    seq_offset_q: Optional[Tensor] = None,
    seq_offset_k: Optional[Tensor] = None,
    num_context: Optional[Tensor] = None,
    num_target: Optional[Tensor] = None,
    target_group_size: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]
```

## 参数说明

### 输入参数

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ---- | --------- | -------- | -------- | -------- | ---- | ---- |
| grad | 输入 | Tensor (REQUIRED) | float32/float16/bf16 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[1, 16]<br>D/D_v∈[16, 512]且是16的倍数 | 前向输出 out 的反向梯度，Jagged 模式下 T 为所有 batch 的序列长度 seq 的和 |
| q | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | B、S、N、D 同 grad<br>D_qk∈[1, 512] | Q 张量 |
| k | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 同 q | K 张量，Shape 和类型与 Q 一致 |
| v | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | 同 grad | V 张量，Shape 和类型与 grad 一致 |
| mask | 输入 | Tensor (OPTIONAL) | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 同 grad | mask 张量，当 mask_type=3 时必须提供 |
| attn_bias | 输入 | Tensor (OPTIONAL) | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 同 grad | attn_bias 张量 |
| seq_offset_q | 输入 | Tensor (OPTIONAL) | int32/int64 | [BatchSize+1] | BatchSize∈[1, 2048] | Jagged 模式下必须提供，表示每个 batch 的序列长度偏移 |
| seq_offset_k | 输入 | Tensor (OPTIONAL) | int32/int64 | [BatchSize+1] | BatchSize∈[1, 2048] | Jagged 模式下必须提供 |
| num_context | 输入 | Tensor (OPTIONAL) | int32/int64 | [B] | B∈[1, 2048]，值范围[1, 128] | Context 掩码长度 |
| num_target | 输入 | Tensor (OPTIONAL) | int32/int64 | [B] | B∈[1, 2048]，值范围[1, 512] | Target 掩码长度 |

### 属性参数

| 名称 | 参数类型 | 数据类型 | 默认值 | 范围/取值 | 说明 |
| ---- | -------- | -------- | ------ | --------- | ---- |
| mask_type | Attr | int | - | [0, 2, 3] | mask 类型：0 表示使用内置下三角 mask（TRIL），2 表示不使用 mask，3 表示使用自定义 mask（CUSTOM）<br>注意：1（TRIU）当前不支持 |
| max_seq_len_q | Attr | int | - | [1, 20480] | 模型最大序列长度 |
| max_seq_len_k | Attr | int | - | [1, 20480] | 模型最大序列长度 |
| silu_scale | Attr | float | 0.0 | 任意值 | SiLU 激活函数前的缩放因子，如果为 0.0 则自动计算为 1.0/max_seq_len |
| target_group_size | Attr (OPTIONAL) | int | None | {1, 3} | Target 分组大小 |
| alpha | Attr (OPTIONAL) | float | 1.0 | 任意值 | 缩放因子 |

### 输出参数

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ---- | --------- | -------- | -------- | -------- | ---- | ---- |
| q_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 与 q 保持一致 | Q 的梯度 |
| k_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 与 k 保持一致 | K 的梯度 |
| v_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | 与 v 保持一致 | V 的梯度 |
| attn_bias_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 与 attn_bias 保持一致 | attn_bias 的反向梯度，如果未提供 attn_bias 则返回空张量 |

## 约束说明

### 算子约束

- 支持 FLOAT32、FLOAT16、BF16 数据类型
- 输入必须为 3D [T, N, D]
- head_dim 必须为 16 的倍数
- mask_type 支持：0 (tril), 2 (none), 3 (custom)
- seq_offset_q 和 seq_offset_k 必须提供，长度为 [B+1]
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 torch_extension 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- mask_type 必须在 [0, 3] 范围内
- max_seq_len 必须在 [1, 20480] 范围内
- 需要安装 `torch_npu`

## 调用示例

参考 [test_torch_hstu_jagged_backward.py](../examples/torch/test_torch_hstu_backward_v1_jagged.py)
