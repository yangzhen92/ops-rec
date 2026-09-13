# `jagged_jagged_bmm_jagged_out_kernel` 算子文档

## 来源

`jagged_jagged_bmm_jagged_out_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_bmm_jagged_out.py>

## 功能说明

`jagged_jagged_bmm_jagged_out_kernel` 用于对两组 Jagged 矩阵逐 batch 执行矩阵乘法，并将每个 batch 的二维结果按行主序展平后拼接为一维 Jagged 输出。

设 batch 数为 `B`，第 `i` 个 batch 的矩阵尺寸为：

```text
A_i   : (m_i, K)
B_i^T : (K, n_i)
C_i   : (m_i, n_i)
```

所有 batch 的输入采用拼接布局：

```text
jagged_A : (sum(m_i), K)
jagged_B : (K, sum(n_i))
jagged_C : (sum(m_i * n_i),)
```

其中 `jagged_B` 已是转置后的布局。对每个 batch，算子根据 offsets 取出对应矩阵并计算：

```text
A_i   = jagged_A[offsets_m[i] : offsets_m[i + 1], :]
B_i^T = jagged_B[:, offsets_n[i] : offsets_n[i + 1]]
C_i   = A_i @ B_i^T
```

随后将 `C_i.reshape(-1)` 写入由 `offsets_mn[i]` 指定的输出段。

### 序列截断

当原始矩阵行列数超过 `max_seq_len` 时，实际参与计算的尺寸为：

```text
M_i = min(m_i, max_seq_len)
N_i = min(n_i, max_seq_len)
```

此时只计算 `A_i[:M_i] @ B_i^T[:, :N_i]`，并将得到的 `M_i * N_i` 个元素紧凑写在第 `i` 个输出段开头。输出段中其余未写入位置保持调用前的值；配套 wrapper 将输出初始化为零，因此这些位置为零。

## 输入

| 参数 | 说明 |
|------|------|
| `jagged_A` | A 矩阵的 Jagged values，shape 为 `(sum(m_i), K)` |
| `jagged_B` | B 矩阵的预转置 Jagged values，shape 为 `(K, sum(n_i))` |
| `lengths_m` | 各 batch 的 `m_i`，shape 为 `(B,)` |
| `lengths_n` | 各 batch 的 `n_i`，shape 为 `(B,)` |
| `lengths_mn` | 各 batch 的输出段长度 `m_i * n_i`，shape 为 `(B,)` |
| `offsets_m` | `lengths_m` 的前缀和，shape 为 `(B + 1,)` |
| `offsets_n` | `lengths_n` 的前缀和，shape 为 `(B + 1,)` |
| `offsets_mn` | `lengths_mn` 的前缀和，shape 为 `(B + 1,)` |
| `max_seq_len` | 每个 batch 参与计算的最大行数和列数 |
| `allow_tf32` | 是否允许 float32 矩阵乘采用 TF32 计算模式，默认值为 `False` |

`jagged_A` 与 `jagged_B` 支持 `fp32`、`fp16` 和 `bf16`。三组 lengths 和 offsets 使用整数类型；本目录封装使用 `int32`。

## 输出

`jagged_C`：一维 tensor，分配长度为 `sum(lengths_mn)`，dtype 与 `jagged_A` 相同。第 `i` 个 batch 的矩阵结果以行主序存放在从 `offsets_mn[i]` 开始的输出段中。

当不存在截断时：

```text
jagged_C[offsets_mn[i] : offsets_mn[i + 1]] = C_i.reshape(-1)
```

## 调用示例

下面在 NPU 上调用配套 wrapper。示例包含两个不同尺寸的 batch，且 `jagged_B` 直接使用预转置布局：

```python
import torch

from triton_jagged_bmm_jagged_out import (
    triton_jagged_jagged_bmm_jagged_out,
)

device = "npu"
K = 4

# 两个 batch：m = [2, 3]，n = [1, 2]
lengths_m = torch.tensor([2, 3], dtype=torch.int32, device=device)
lengths_n = torch.tensor([1, 2], dtype=torch.int32, device=device)
lengths_mn = lengths_m * lengths_n

offsets_m = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
offsets_n = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
offsets_mn = torch.tensor([0, 2, 8], dtype=torch.int32, device=device)

jagged_A = torch.randn((5, K), dtype=torch.float32, device=device)
jagged_B = torch.randn((K, 3), dtype=torch.float32, device=device)

jagged_C = triton_jagged_jagged_bmm_jagged_out(
    jagged_A=jagged_A,
    jagged_B=jagged_B,
    max_seq_len=3,
    lengths_m=lengths_m,
    lengths_n=lengths_n,
    lengths_mn=lengths_mn,
    offsets_m=offsets_m,
    offsets_n=offsets_n,
    offsets_mn=offsets_mn,
    allow_tf32=False,
)

expected = torch.cat(
    [
        (jagged_A[0:2] @ jagged_B[:, 0:1]).reshape(-1),
        (jagged_A[2:5] @ jagged_B[:, 1:3]).reshape(-1),
    ]
)

torch.testing.assert_close(jagged_C, expected)
print(jagged_C.shape)  # torch.Size([8])
```

## 关键约束

- `jagged_A` 和 `jagged_B` 必须是二维 tensor，且共享维满足 `jagged_A.shape[1] == jagged_B.shape[0] == K`。
- `jagged_A`、`jagged_B` 必须位于同一 device，并使用相同 dtype；通常应传入 contiguous tensor。
- `lengths_m`、`lengths_n` 和 `lengths_mn` 的长度必须相同，均等于 batch 数 `B`。
- `lengths_mn[i]` 必须等于 `lengths_m[i] * lengths_n[i]`。
- 三组 offsets 必须是长度为 `B + 1` 的 contiguous 一维整数 tensor，首元素为零且单调不减。
- `offsets_m[-1]` 必须等于 `jagged_A.shape[0]`，`offsets_n[-1]` 必须等于 `jagged_B.shape[1]`。
- `offsets_mn[-1]` 必须等于 `sum(lengths_mn)`，并与输出长度一致。
- `max_seq_len` 必须为正数。若需要计算完整矩阵，应满足 `max_seq_len >= max(max(lengths_m), max(lengths_n))`。
- 配套 wrapper 以零初始化输出；发生截断时，各输出段内未被写入的位置保持为零。

## ATK 测试

以下命令约定 NPU 端已部署 `npu_atk_test/` 目录及对应 YAML、生成器、API 和 JSON 文件。当前工作区中的该算子目录仅包含 `gpu_atk_test/`，如 NPU 侧文件尚未同步，需要先完成部署。

生成及测试命令均在 NPU 端的 `npu_atk_test` 目录下执行：

```bash
cd experimental/triton/atk_test/jagged_jagged_bmm_jagged_out_kernel/npu_atk_test
```

### 生成测试数据

```bash
atk case -f jagged_jagged_bmm_jagged_out_kernel.yaml \
    -p generate_jagged_jagged_bmm_jagged_out_kernel.py
```

生成的用例配置文件为 `all_jagged_jagged_bmm_jagged_out_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_jagged_jagged_bmm_jagged_out_kernel.json --task accuracy \
    -p jagged_jagged_bmm_jagged_out_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_jagged_jagged_bmm_jagged_out_kernel.json \
    --task performance_device \
    -p jagged_jagged_bmm_jagged_out_kernel_api.py
```
