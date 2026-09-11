# `_jagged_flash_attention_bwd_basic_kernel` 算子文档

## 来源

`_jagged_flash_attention_bwd_basic_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/fe980ab54a6e28818d81c8694b6564e7f804418b/fbgemm_gpu/fbgemm_gpu/sll/triton_sll.py#L2034-L2247>

## 功能说明

`_jagged_flash_attention_bwd_basic_kernel` 是基础版 Jagged Flash Attention 的反向算子，用于根据 Query、Key、Value、上游梯度及前向阶段保存的中间量，计算 `dq`、`dk` 和 `dv`。

输入 token 采用二维展平布局，`offset` 将它们划分为多个变长序列。算子在每个序列内部独立执行反向计算，不会跨序列传播或累加梯度。与 multi-head 版本不同，该算子的输入不包含注意力头维度。

设：

- `T` 为所有 Jagged 序列的 token 总数；
- `D` 为每个 token 的特征维度；
- `C = max_seq_len`；
- 第 `b` 个序列参与计算的长度为

```text
M_b = min(offset[b + 1] - offset[b], C)
```

对任一序列，定义：

```text
S     = Q @ K^T
P     = exp(S - LSE[:, None]) / C
G     = dO @ V^T
```

当 `use_mask=False` 时，所有 query-key 位置均参与注意力；当 `use_mask=True` 时，掩码保留 `query_index <= key_index` 的上三角区域。令 `R` 表示该掩码：

```text
R[i, j] = 1                              use_mask=False
R[i, j] = 1 if i <= j else 0             use_mask=True
```

最终梯度为：

```text
dS = P * (R * G - C * delta[:, None])

dQ = dS @ K
dK = dS^T @ Q
dV = (P * R)^T @ dO
```

其中前向保存的中间量满足：

```text
LSE[i]   = logsumexp(S[i, :])
O        = (P * R) @ V
delta[i] = sum(O[i, d] * dO[i, d] for d in range(D))
```

需要注意，`LSE` 对完整的 `S` 行进行归一化；启用 `use_mask` 时，掩码应用在注意力值上，而不是在计算 `LSE` 前将分数置为负无穷。注意力分数也不包含常见的 `1 / sqrt(D)` 缩放。

## 输入

| 参数 | 说明 |
|------|------|
| `q` | Query，shape 为 `(T, D)` |
| `k` | Key，shape 为 `(T, D)`，以非转置形式输入 |
| `v` | Value，shape 为 `(T, D)` |
| `o` | 配套前向算子的输出，shape 为 `(T, D)`；用于预处理得到 `delta` |
| `do` | 输出 `o` 的上游梯度，shape 为 `(T, D)` |
| `delta` | 反向预处理产生的逐 token 点积，shape 为 `(T,)`，满足 `delta = sum(o * do, dim=-1)` |
| `lse` | 前向阶段保存的 log-sum-exp，shape 为 `(T,)` |
| `offset` | Jagged 序列偏移，shape 为 `(B + 1,)`；第 `b` 个序列对应 `[offset[b], offset[b + 1])` |
| `max_seq_len` | 单个序列参与计算的最大 token 数，同时参与注意力概率和梯度缩放 |
| `use_mask` | 是否应用 `query_index <= key_index` 的上三角注意力掩码 |
| `allow_tf32` | 是否允许矩阵乘采用 TF32 计算模式 |
| `D` | token 的特征维度，应等于 `q.shape[1]` |

`q`、`k`、`v`、`o`、`do`、`delta` 和 `lse` 支持 `fp32`、`fp16` 和 `bf16`，`offset` 使用 `int64`。

## 输出

| 输出 | 说明 |
|------|------|
| `dq` | Query 梯度，shape 为 `(T, D)` |
| `dk` | Key 梯度，shape 为 `(T, D)` |
| `dv` | Value 梯度，shape 为 `(T, D)` |

对于长度超过 `max_seq_len` 的序列，仅前 `max_seq_len` 个 token 对应的梯度会被写入。当前目录封装将三个输出初始化为零，因此被截断的位置保持为零。

当前 ATK API 以 `dq` 作为任务输出进行两端对比；kernel 本身同时计算并写入 `dq`、`dk` 和 `dv`。

## 计算示例

下面使用 NPU 上的 PyTorch 运算表达 `use_mask=True` 时的算子语义，不涉及具体的 Kernel 启动参数：

```python
import torch

device = "npu"
T, D = 5, 32
max_seq_len = 3
use_mask = True

q = torch.randn(T, D, dtype=torch.float32, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
do = torch.randn_like(q)

# 两个 Jagged 序列，长度分别为 3 和 2
offset = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)

o = torch.zeros_like(q)
lse = torch.zeros(T, dtype=q.dtype, device=device)
delta = torch.zeros_like(lse)
dq = torch.zeros_like(q)
dk = torch.zeros_like(k)
dv = torch.zeros_like(v)

for b in range(offset.numel() - 1):
    begin = int(offset[b].item())
    end = min(int(offset[b + 1].item()), begin + max_seq_len)
    if begin == end:
        continue

    q_b = q[begin:end]
    k_b = k[begin:end]
    v_b = v[begin:end]
    do_b = do[begin:end]

    scores = torch.matmul(q_b, k_b.transpose(0, 1))
    lse_b = torch.logsumexp(scores, dim=-1)
    p = torch.exp(scores - lse_b[:, None]) / max_seq_len

    length = end - begin
    positions = torch.arange(length, device=device)
    keep = positions[:, None] <= positions[None, :]
    p_masked = torch.where(keep, p, 0.0)

    o_b = torch.matmul(p_masked, v_b)
    delta_b = torch.sum(o_b * do_b, dim=-1)
    g = torch.matmul(do_b, v_b.transpose(0, 1))
    g_masked = torch.where(keep, g, 0.0)
    ds = p * (g_masked - max_seq_len * delta_b[:, None])

    o[begin:end] = o_b
    lse[begin:end] = lse_b
    delta[begin:end] = delta_b
    dq[begin:end] = torch.matmul(ds, k_b)
    dk[begin:end] = torch.matmul(ds.transpose(0, 1), q_b)
    dv[begin:end] = torch.matmul(p_masked.transpose(0, 1), do_b)

print(dq.shape, dk.shape, dv.shape)
# torch.Size([5, 32]) torch.Size([5, 32]) torch.Size([5, 32])
```

## 关键约束

- `q`、`k`、`v`、`o` 和 `do` 必须是二维 tensor，shape 均为 `(T, D)`。
- `delta` 和 `lse` 必须是一维 tensor，长度均为 `T`。
- 所有浮点输入必须位于同一 device，并具有相互匹配的 dtype 和内存布局；通常应传入 contiguous tensor。
- `offset` 必须是一维 `int64` tensor，满足 `offset[0] = 0`、`offset[-1] = T`，并且单调不减。
- `max_seq_len` 必须为正数。超过该长度的序列会被截断，截断范围外的梯度不会被写入。
- `delta` 必须由与本次反向计算对应的 `o` 和 `do` 生成，且 `o` 必须采用相同的 `max_seq_len` 和 `use_mask`。
- `lse` 必须来自同一组 `q`、`k`、`offset` 和 `max_seq_len` 的前向计算；其归一化本身不应用 `use_mask`。
- `use_mask=True` 保留的是上三角区域 `i <= j`，不能将其当作保留下三角区域的常见 causal mask。
- 算子不包含 dropout，也不接受额外的 attention mask。

## ATK 测试

以下命令在 NPU 端的 `npu_atk_test` 目录下执行：

```bash
cd experimental/triton/atk_test/_jagged_flash_attention_bwd_basic_kernel/npu_atk_test
```

### 生成测试数据

```bash
atk case -f _jagged_flash_attention_bwd_basic_kernel.yaml \
    -p generate__jagged_flash_attention_bwd_basic_kernel.py
```

生成的用例配置文件为 `all__jagged_flash_attention_bwd_basic_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__jagged_flash_attention_bwd_basic_kernel.json --task accuracy \
    -p _jagged_flash_attention_bwd_basic_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__jagged_flash_attention_bwd_basic_kernel.json \
    --task performance_device \
    -p _jagged_flash_attention_bwd_basic_kernel_api.py
```
