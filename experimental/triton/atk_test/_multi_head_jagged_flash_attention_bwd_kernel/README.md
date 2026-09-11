# `_multi_head_jagged_flash_attention_bwd_kernel` 算子文档

## 来源

`_multi_head_jagged_flash_attention_bwd_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/fe980ab54a6e28818d81c8694b6564e7f804418b/fbgemm_gpu/fbgemm_gpu/sll/triton_sll.py#L3501-L3759>

## 功能说明

`_multi_head_jagged_flash_attention_bwd_kernel` 是多头 Jagged Flash Attention 的反向算子，用于根据前向阶段保存的中间量和输出梯度，计算 Query、Key、Value 对应的梯度 `dq`、`dk` 和 `dv`。

输入 token 以展平形式存储，`offset` 将它们划分为多个变长序列。算子在每个序列、每个注意力头内独立完成反向计算，不会在不同序列或不同注意力头之间混合梯度。

设：

- `H` 为注意力头数；
- `T` 为所有序列的 token 总数；
- `D` 为单个注意力头的特征维度；
- `C = max_seq_len`；
- 第 `b` 个序列参与计算的长度为

```text
L_b = min(offset[b + 1] - offset[b], C)
```

对于第 `b` 个序列和第 `h` 个注意力头，反向计算可表示为：

```text
S     = Q @ K^T
P     = exp(S - LSE[:, None]) / C
dP    = dO @ V^T
dS    = P * (dP - C * delta[:, None])

dQ    = dS @ K
dK    = dS^T @ Q
dV    = P^T @ dO
```

其中：

```text
LSE[i]   = logsumexp(S[i, :])
delta[i] = sum(O[i, d] * dO[i, d] for d in range(D))
```

`P` 对应前向 softmax 概率除以 `max_seq_len` 后的结果。该公式与配套前向算子的语义一致：注意力分数不包含 `1 / sqrt(D)` 缩放，前向输出包含额外的 `1 / max_seq_len` 缩放。

## 输入

| 参数 | 说明 |
|------|------|
| `q` | Query，shape 为 `(H, T, D)` |
| `k` | Key，shape 为 `(H, T, D)` |
| `v` | Value，shape 为 `(H, T, D)` |
| `o` | 配套前向算子的输出，shape 为 `(H, T, D)`；用于预处理得到 `delta`，主反向计算直接使用 `delta` |
| `do` | 输出 `o` 的上游梯度，shape 为 `(H, T, D)` |
| `delta` | 反向预处理中得到的逐 token 点积，shape 为 `(H, T)`，满足 `delta = sum(o * do, dim=-1)` |
| `lse` | 前向阶段保存的 log-sum-exp，shape 为 `(H, T)` |
| `offset` | Jagged 序列偏移，shape 为 `(B + 1,)`；第 `b` 个序列对应 `[offset[b], offset[b + 1])` |
| `num_heads` | 注意力头数，应等于 `q.shape[0]` |
| `max_seq_len` | 每个序列参与计算的最大 token 数，同时参与注意力概率和梯度的缩放 |
| `D` | 单个注意力头的特征维度，应等于 `q.shape[2]` |
| `allow_tf32` | 是否允许矩阵乘采用 TF32 计算模式 |

`q`、`k`、`v`、`o`、`do`、`delta` 和 `lse` 支持 `fp32`、`fp16` 和 `bf16`，`offset` 使用 `int64`。

## 输出

| 输出 | 说明 |
|------|------|
| `dq` | Query 梯度，shape 为 `(H, T, D)` |
| `dk` | Key 梯度，shape 为 `(H, T, D)` |
| `dv` | Value 梯度，shape 为 `(H, T, D)` |

对于长度超过 `max_seq_len` 的序列，仅序列前 `max_seq_len` 个 token 对应的梯度会被写入。当前目录封装将三个输出初始化为零，因此被截断的位置保持为零。

当前 ATK API 以 `dq` 作为任务输出进行两端对比；kernel 本身同时计算并写入 `dq`、`dk` 和 `dv`。

## 计算示例

下面使用 NPU 上的 PyTorch 运算表达与算子一致的数学语义，不涉及具体的 Kernel 启动参数：

```python
import torch

device = "npu"
H, T, D = 2, 5, 32
max_seq_len = 3

q = torch.randn(H, T, D, dtype=torch.float32, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
do = torch.randn_like(q)

# 两个 Jagged 序列，长度分别为 3 和 2
offset = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)

o = torch.zeros_like(q)
lse = torch.zeros((H, T), dtype=q.dtype, device=device)
delta = torch.zeros_like(lse)
dq = torch.zeros_like(q)
dk = torch.zeros_like(k)
dv = torch.zeros_like(v)

for b in range(offset.numel() - 1):
    begin = int(offset[b].item())
    end = min(int(offset[b + 1].item()), begin + max_seq_len)
    if begin == end:
        continue

    q_b = q[:, begin:end, :]
    k_b = k[:, begin:end, :]
    v_b = v[:, begin:end, :]
    do_b = do[:, begin:end, :]

    scores = torch.matmul(q_b, k_b.transpose(-1, -2))
    lse_b = torch.logsumexp(scores, dim=-1)
    p = torch.exp(scores - lse_b.unsqueeze(-1)) / max_seq_len
    o_b = torch.matmul(p, v_b)
    delta_b = torch.sum(o_b * do_b, dim=-1)

    dp = torch.matmul(do_b, v_b.transpose(-1, -2))
    ds = p * (dp - max_seq_len * delta_b.unsqueeze(-1))

    o[:, begin:end, :] = o_b
    lse[:, begin:end] = lse_b
    delta[:, begin:end] = delta_b
    dq[:, begin:end, :] = torch.matmul(ds, k_b)
    dk[:, begin:end, :] = torch.matmul(ds.transpose(-1, -2), q_b)
    dv[:, begin:end, :] = torch.matmul(p.transpose(-1, -2), do_b)

print(dq.shape, dk.shape, dv.shape)
# torch.Size([2, 5, 32]) torch.Size([2, 5, 32]) torch.Size([2, 5, 32])
```

## 关键约束

- `q`、`k`、`v`、`o` 和 `do` 必须是三维 tensor，shape 均为 `(H, T, D)`。
- `delta` 和 `lse` 必须是二维 tensor，shape 均为 `(H, T)`。
- 所有浮点输入必须位于同一 device，并具有相互匹配的 dtype 和内存布局；通常应传入 contiguous tensor。
- `offset` 必须是一维 `int64` tensor，满足 `offset[0] = 0`、`offset[-1] = T`，并且单调不减。
- `num_heads` 必须等于 `H`，`D` 必须等于输入张量的最后一维。
- `max_seq_len` 必须为正数。超过该长度的序列会被截断，截断范围外的梯度不会被写入。
- `delta` 必须由与本次反向计算对应的 `o` 和 `do` 生成，`lse` 必须来自同一组 `q`、`k`、`offset` 和 `max_seq_len` 的前向计算。
- 算子不包含 causal mask、外部 attention mask 或 dropout，其反向语义必须与配套前向算子保持一致。
- 空序列可以用相邻且相等的 offset 表示，该序列不会产生梯度输出。

## ATK 测试

以下命令在 NPU 端的 `npu_atk_test` 目录下执行：

```bash
cd experimental/triton/atk_test/_multi_head_jagged_flash_attention_bwd_kernel/npu_atk_test
```

### 生成测试数据

```bash
atk case -f _multi_head_jagged_flash_attention_bwd_kernel.yaml \
    -p generate__multi_head_jagged_flash_attention_bwd_kernel.py
```

生成的用例配置文件为 `all__multi_head_jagged_flash_attention_bwd_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__multi_head_jagged_flash_attention_bwd_kernel.json --task accuracy \
    -p _multi_head_jagged_flash_attention_bwd_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__multi_head_jagged_flash_attention_bwd_kernel.json \
    --task performance_device \
    -p _multi_head_jagged_flash_attention_bwd_kernel_api.py
```
