# `_multi_head_jagged_flash_attention_fwd_kernel` 算子文档

## 来源

`_multi_head_jagged_flash_attention_fwd_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/fe980ab54a6e28818d81c8694b6564e7f804418b/fbgemm_gpu/fbgemm_gpu/sll/triton_sll.py#L3260-L3392>

## 功能说明

`_multi_head_jagged_flash_attention_fwd_kernel` 是多头 Jagged Self-Attention 的前向算子。它通过 `offset` 将展平存储的 token 划分为多个变长序列，并在每个序列、每个注意力头内独立计算注意力，不会在不同序列或不同注意力头之间混合数据。

设：

- `H` 为注意力头数；
- `T` 为所有序列的 token 总数；
- `D` 为单个注意力头的特征维度；
- 第 `b` 个序列的有效长度为

```text
L_b = min(offset[b + 1] - offset[b], max_seq_len)
```

对第 `b` 个序列和第 `h` 个注意力头，取该序列前 `L_b` 个 token，计算：

```text
S[b, h]   = Q[b, h] @ K[b, h]^T
P[b, h]   = softmax(S[b, h], dim=-1)
O[b, h]   = (P[b, h] @ V[b, h]) / max_seq_len
LSE[b, h] = logsumexp(S[b, h], dim=-1)
```

其中 `O` 为注意力结果，`LSE` 为每个 query 行对应的 log-sum-exp，供后续反向计算使用。

该算子的语义需要特别注意：

- 注意力分数为原始的 `Q @ K^T`，不包含常见的 `1 / sqrt(D)` 缩放；
- 注意力结果会额外除以 `max_seq_len`；
- 不包含 causal mask、外部 attention mask 或 dropout；
- 当某个序列长度超过 `max_seq_len` 时，只处理该序列的前 `max_seq_len` 个 token。

## 输入

| 参数 | 说明 |
|------|------|
| `q` | Query，shape 为 `(H, T, D)` |
| `k` | Key，shape 为 `(H, T, D)`，shape、dtype 和 device 与 `q` 一致 |
| `v` | Value，shape 为 `(H, T, D)`，shape、dtype 和 device 与 `q` 一致 |
| `offset` | Jagged 序列偏移，shape 为 `(B + 1,)`；第 `b` 个序列对应 token 区间 `[offset[b], offset[b + 1])` |
| `num_heads` | 注意力头数，应等于 `q.shape[0]` |
| `max_seq_len` | 单个序列参与计算的最大 token 数，同时也是输出结果的归一化因子 |
| `D` | 单个注意力头的特征维度，应等于 `q.shape[2]` |
| `allow_tf32` | 是否允许 float32 矩阵乘使用 TF32 计算模式 |

`q`、`k`、`v` 支持 `fp32`、`fp16` 和 `bf16`。

## 输出

| 输出 | 说明 |
|------|------|
| `o` | 注意力结果，shape 为 `(H, T, D)`，dtype 与 `q` 相同 |
| `lse` | 每个 query 行的 log-sum-exp，shape 为 `(H, T)`；本目录封装中 dtype 与 `q` 相同 |

对于长度超过 `max_seq_len` 的序列，超出截断范围的 token 不会被 kernel 写入。本目录封装将 `o` 和 `lse` 初始化为零，因此这些位置保持为零。

## 计算示例

下面使用 PyTorch 表达与算子一致的数学语义，不涉及具体的 Kernel 启动参数：

```python
import torch

device = "cuda"
H, T, D = 2, 5, 32
max_seq_len = 3

q = torch.randn(H, T, D, dtype=torch.float32, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)

# 两个变长序列，长度分别为 3 和 2
offset = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)

o = torch.zeros_like(q)
lse = torch.zeros((H, T), dtype=q.dtype, device=device)

for b in range(offset.numel() - 1):
    begin = int(offset[b].item())
    end = min(int(offset[b + 1].item()), begin + max_seq_len)
    if begin == end:
        continue

    q_b = q[:, begin:end, :]
    k_b = k[:, begin:end, :]
    v_b = v[:, begin:end, :]

    scores = torch.matmul(q_b, k_b.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)

    o[:, begin:end, :] = torch.matmul(probs, v_b) / max_seq_len
    lse[:, begin:end] = torch.logsumexp(scores, dim=-1)

print(o.shape)    # torch.Size([2, 5, 32])
print(lse.shape)  # torch.Size([2, 5])
```

## 关键约束

- `q`、`k`、`v` 必须是三维 tensor，且 shape、dtype 和 device 完全一致。
- `offset` 必须是一维整数 tensor，长度为 `B + 1`，满足 `offset[0] = 0`、`offset[-1] = T`，并且单调不减。
- `num_heads` 必须等于 `q.shape[0]`，`D` 必须等于 `q.shape[2]`。
- `max_seq_len` 必须为正数。序列长度超过该值时，仅序列前缀参与计算，超出部分不会被写入。
- `max_seq_len` 不仅控制截断，还直接参与输出缩放；即使所有序列均短于该值，修改它也会改变 `o` 的数值。
- 算子仅在同一 Jagged 序列、同一注意力头内执行全量双向注意力，不提供 causal mask、外部 attention mask 或 dropout。
- 使用 `allow_tf32=True` 时，GPU 上的 float32 矩阵乘可能采用 TF32，结果会存在相应的数值精度差异。
