# `jagged_jagged_bmm_jagged_out_kernel` 算子文档

## 来源

`jagged_jagged_bmm_jagged_out_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_bmm_jagged_out.py>

## 功能说明

`jagged_jagged_bmm_jagged_out_kernel` 用于对两组 jagged 矩阵逐 batch 执行矩阵乘法，并将每个 batch 的结果按行主序展平后拼接成一维 jagged 输出。

设 batch 数为 `B`，第 `i` 个 batch 的矩阵尺寸为：

```text
A_i   : (m_i, K)
B_i^T : (K, n_i)
C_i   : (m_i, n_i)
```

输入在内存中的组织方式为：

```text
jagged_A : (sum(m_i), K)
jagged_B : (K, sum(n_i))
jagged_C : (sum(m_i * n_i),)
```

其中 `jagged_B` 已经是转置后的布局。对每个 batch，kernel 计算：

```text
A_i = jagged_A[offsets_m[i] : offsets_m[i + 1], :]
B_i^T = jagged_B[:, offsets_n[i] : offsets_n[i + 1]]
C_i = A_i @ B_i^T
```

随后将 `C_i.reshape(-1)` 写入：

```text
jagged_C[offsets_mn[i] : offsets_mn[i + 1]]
```

kernel 内部以 fp32 累加 `tl.dot` 的结果，再转换为输出 tensor 的 dtype 写回。

## 输入

原生 wrapper `triton_jagged_jagged_bmm_jagged_out` 接收以下参数：

| 参数 | 说明 |
|------|------|
| `jagged_A` | A 矩阵的 jagged values，shape `(sum(m_i), K)` |
| `jagged_B` | B 矩阵的转置布局，shape `(K, sum(n_i))` |
| `max_seq_len` | kernel 覆盖的最大序列长度；每段实际使用 `min(length, max_seq_len)` |
| `lengths_m` | 各 batch 的 `m_i`，shape `(B,)`，dtype 为 `int32` |
| `lengths_n` | 各 batch 的 `n_i`，shape `(B,)`，dtype 为 `int32` |
| `lengths_mn` | 各 batch 的输出元素数 `m_i * n_i`，shape `(B,)`，dtype 为 `int32` |
| `offsets_m` | `lengths_m` 的前缀和，shape `(B+1,)`，dtype 为 `int32` |
| `offsets_n` | `lengths_n` 的前缀和，shape `(B+1,)`，dtype 为 `int32` |
| `offsets_mn` | `lengths_mn` 的前缀和，shape `(B+1,)`，dtype 为 `int32` |
| `allow_tf32` | 是否允许 fp32 矩阵乘法使用 TF32，默认 `False` |

YAML 用例只显式提供 `jagged_A`、`jagged_B`、三组 offsets 和 `max_seq_len`。`lengths_m`、`lengths_n`、`lengths_mn` 由 ATK API 在预处理阶段构造。

## 输出

`jagged_C`：一维 tensor，长度为 `sum(m_i * n_i)`，dtype 与 `jagged_A` 相同。第 `i` 段保存 `C_i` 的行主序展平结果。

当前 ATK 用例约束 `n_i = m_i`，因此输出长度为：

```text
sum(m_i^2 for i in range(B))
```

## ATK 输入重建规则

YAML/JSON 中的 jagged 元信息不能直接表达每个 batch 的完整变长序列，所以当前 GPU/NPU API 使用相同的确定性规则重建输入关系：

1. 由 `offsets_m` 输入的长度确定 batch 数：`B = offsets_m.shape[0] - 1`。原始 offset 数值不参与计算。
2. 从 `jagged_A.shape` 取得 `total_m` 和 `K`，要求 `total_m >= B`。
3. 使用只依赖 `(total_m, B)` 的确定性 LCG 权重，将 `total_m` 切分为 `B` 个非空、变长的 `m_i`。
4. 设置 `n_i = m_i`，并计算 `lengths_mn[i] = m_i * n_i`。
5. 分别对三组 lengths 做前缀和，重建 `offsets_m`、`offsets_n` 和 `offsets_mn`。
6. 设置 `max_seq_len = max(lengths_m)`，保证当前 ATK 测试不截断任何 batch。

因此，YAML/JSON 中 `offsets_n`、`offsets_mn` 和 `max_seq_len` 的具体值不会直接传给 wrapper；`offsets_m` 也只使用 shape。生成器负责保证：

```text
jagged_A.shape == (total_m, K)
jagged_B.shape == (K, total_m)
offsets_m.shape == offsets_n.shape == offsets_mn.shape == (B + 1,)
jagged_A.dtype == jagged_B.dtype
```

## dtype 适配

YAML 支持 `fp32`、`fp16` 和 `bf16`。当前 GPU/NPU ATK API 会记录 `jagged_A` 的原始 dtype，将 `jagged_A` 和 `jagged_B` 转为 fp32 后执行 kernel，再把 `jagged_C` 转回原始 dtype。

这样可以避免 Ascend 编译链将 `tl.load(..., other=0.0)` 中的字面量提升为 fp64 后导致 `tl.dot` 编译失败，同时确保 GPU 与 NPU 返回相同的输出 dtype。ATK 调用 wrapper 时固定使用 `allow_tf32=False`。

## 调用示例

通过 Python 直接调用原生 wrapper（以下代码在 `gpu_atk_test/` 或 `npu_atk_test/` 目录中执行）：

```python
import torch

from triton_jagged_bmm_jagged_out import (
    triton_jagged_jagged_bmm_jagged_out,
)

device = "npu"
K = 4

# 两个 batch：m = n = [2, 3]
lengths_m = torch.tensor([2, 3], dtype=torch.int32, device=device)
lengths_n = torch.tensor([2, 3], dtype=torch.int32, device=device)
lengths_mn = lengths_m * lengths_n

offsets_m = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
offsets_n = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
offsets_mn = torch.tensor([0, 4, 13], dtype=torch.int32, device=device)

jagged_A = torch.randn((5, K), dtype=torch.float32, device=device)
jagged_B = torch.randn((K, 5), dtype=torch.float32, device=device)

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
        (jagged_A[0:2] @ jagged_B[:, 0:2]).reshape(-1),
        (jagged_A[2:5] @ jagged_B[:, 2:5]).reshape(-1),
    ]
)
torch.testing.assert_close(jagged_C, expected)
print(jagged_C.shape)  # torch.Size([13])
```

如在 CUDA GPU 上运行，将 `device` 改为 `"cuda"`。

## Kernel 启动配置

GPU 与 NPU 版本均使用固定分块：

```text
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
```

wrapper 根据 `max_seq_len` 和 batch 数构造二维 grid：

```text
num_blocks_m = ceil_div(max_seq_len, BLOCK_SIZE_M)
num_blocks_n = ceil_div(max_seq_len, BLOCK_SIZE_N)
grid = (num_blocks_m * num_blocks_n, B)
```

grid 第一维枚举每个 batch 内的 `(M, N)` tile，第二维枚举 batch。超过当前 batch 实际 `m_i` 或 `n_i` 的 program 会直接返回，K 维尾块通过 mask 补零。

## 关键约束

- `jagged_A` 和 `jagged_B` 必须是二维 tensor，且共享维满足 `jagged_A.shape[1] == jagged_B.shape[0]`。
- `jagged_A`、`jagged_B` 必须位于同一设备并使用相同 dtype；当前 YAML 支持 fp32、fp16 和 bf16。
- `offsets_m`、`offsets_n` 和 `offsets_mn` 必须是 contiguous 的一维 `int32` tensor，首元素为 0 且单调递增。
- 三组 lengths 和 offsets 必须一致：`offsets_m[-1] = sum(m_i)`、`offsets_n[-1] = sum(n_i)`、`offsets_mn[-1] = sum(m_i * n_i)`。
- `jagged_A.shape[0] == offsets_m[-1]`，`jagged_B.shape[1] == offsets_n[-1]`。
- 原生 kernel 会将每个 batch 的 `m_i`、`n_i` 截断到 `max_seq_len`；如果需要完整结果，应满足 `max_seq_len >= max(max(lengths_m), max(lengths_n))`。当前 ATK API 自动满足该条件。
- 当前 ATK 用例要求 `B > 0`、`total_m >= B`，并构造每个 batch 至少包含一行的 `m_i = n_i`。
- `jagged_C` 由 wrapper 以零初始化；被 `max_seq_len` 截断而未计算的位置保持为零。

## 目录结构

```text
experimental/triton/atk_test/jagged_jagged_bmm_jagged_out_kernel/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_bmm_jagged_out.py              # GPU(CUDA) 端 kernel 与 wrapper
│   └── jagged_jagged_bmm_jagged_out_kernel_api.py   # GPU 端 ATK API
└── npu_atk_test/
    ├── triton_jagged_bmm_jagged_out.py              # NPU 端 kernel 与 wrapper
    ├── jagged_jagged_bmm_jagged_out_kernel.yaml     # ATK 用例设计
    ├── generate_jagged_jagged_bmm_jagged_out_kernel.py # shape/dtype 约束生成器
    ├── jagged_jagged_bmm_jagged_out_kernel_api.py   # NPU 端 ATK API
    └── all_jagged_jagged_bmm_jagged_out_kernel.json # 已生成的测试用例
```

## ATK 测试

测试在 GPU 与 NPU 两端协同进行：`gpu_atk_test/` 部署到 GPU 端，`npu_atk_test/` 部署到 NPU 端。以下生成及测试命令均在 NPU 端的 `npu_atk_test/` 目录中执行：

```bash
cd experimental/triton/atk_test/jagged_jagged_bmm_jagged_out_kernel/npu_atk_test
```

### 远端 GPU 服务器启动监听

在 GPU 端进入 `gpu_atk_test/` 目录后执行：

```bash
atk server --devices 0 \
    --plugin_path jagged_jagged_bmm_jagged_out_kernel_api.py
```

如 GPU server 不在默认地址或端口，请在 NPU 端执行测试时按 ATK 环境配置远端 host/port。

### 生成测试数据

```bash
atk case \
    -f jagged_jagged_bmm_jagged_out_kernel.yaml \
    -p generate_jagged_jagged_bmm_jagged_out_kernel.py
```

生成的用例通常位于：

```text
result/jagged_jagged_bmm_jagged_out_kernel/json/
all_jagged_jagged_bmm_jagged_out_kernel.json
```

本目录同时提供可直接复用的 `all_jagged_jagged_bmm_jagged_out_kernel.json`。

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_jagged_jagged_bmm_jagged_out_kernel.json \
    --task accuracy \
    -p jagged_jagged_bmm_jagged_out_kernel_api.py
```

如需保存精度对比数据，在命令末尾增加 `--save_data profile`。

### 性能测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_jagged_jagged_bmm_jagged_out_kernel.json \
    --task performance_device \
    -p jagged_jagged_bmm_jagged_out_kernel_api.py
```

如需保存性能对比数据，在命令末尾增加 `--save_data profile`。
