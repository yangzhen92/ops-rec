# triton_add_position_embeddings_kernel

## 来源

`add_position_embeddings` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 功能说明

triton_add_position_embeddings_kernel 是一个基于 Triton 实现的 Ascend NPU 自定义算子，用于将位置编码（Dense 查找表）按 batch 叠加到 jagged 输入上：

- 输入 `Jagged` 形状为 `(sum_B(N_i), D)`，`Dense` 位置编码表形状为 `(K, D)`
- 对每个 batch `b`（行区间 `[seq_offsets[b], seq_offsets[b+1])`）内的第 `n` 行：
  - 计算查找索引 `idx = min(n + ind_offsets[b], high_inds[b])`（无 `ind_offsets` 时为 `min(n, high_inds[b])`）
  - 输出 `Out[n] = Jagged[n] * scale（SCALE_JAGGED=true 时）+ Dense[idx]`
- 输出 `Out` 形状为 `(sum_B(N_i), D)`

### 输入

- `Jagged` `(sum_B(N_i), D)` — jagged 拼接后的输入
- `seq_offsets` `(B+1,)` — 每个 batch 的起止行偏移
- `high_inds` `(B,)` — 每个 batch 的 dense 表最大 lookup 索引
- `Dense` `(K, D)` — 位置编码查找表
- `scale`（标量，默认 `1.0`）— 缩放系数
- `ind_offsets` `(B,)`（可选，默认 None）— 每个 batch 的索引偏移
- `SCALE_JAGGED`（attr bool，默认 false）— 为 true 时先对 Jagged 乘以 scale

### 输出

- `Out` `(sum_B(N_i), D)`，与 `Jagged` 同形状

### 约束

- `seq_offsets.shape[0] = B + 1`，`high_inds.shape[0] = B`，`ind_offsets.shape[0] = B`（存在时）
- `Jagged.shape[0] = seq_offsets[-1]`（总行数 = 最后一个 offset）
- `Dense.shape[1] = Jagged.shape[1] = D`
- `high_inds[i] <= K - 1`（Dense 表索引范围 0 ~ K-1）
- 数据类型：fp32 / fp16 / bf16

## 目录结构

```text
experimental/atk_test/triton_add_position_embeddings_kernel/
├── README.md
├── gpu_atk_test/
│   ├── triton_add_position_embeddings_kernel.py   # GPU 端 Triton kernel 实现（含 triton_autotune）
│   └── add_position_embeddings_api.py             # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── triton_add_position_embeddings_kernel.py   # NPU 端 Triton kernel 实现与对外 Python API
    ├── add_position_embeddings.yaml               # ATK 用例设计（dtype/shape/约束）
    ├── generate_add_position_embeddings.py        # ATK 用例生成器（保证 B/总行数/D 维度一致）
    ├── add_position_embeddings_api.py             # NPU 端 ATK API 执行文件
    └── all_add_position_embeddings.json           # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用：

```python
import torch

from triton_add_position_embeddings_kernel import add_position_embeddings

# case：B=2 个 batch，D=64，Dense 表 K=64 行
seq_offsets = torch.tensor([0, 100, 200], dtype=torch.int32, device="npu:2")
high_inds = torch.tensor([30, 60], dtype=torch.int32, device="npu:2")
jagged = torch.randn(200, 64, dtype=torch.bfloat16, device="npu:2")
dense = torch.randn(64, 64, dtype=torch.bfloat16, device="npu:2")

output = add_position_embeddings(jagged, seq_offsets, high_inds, dense)
print(output.shape)  # torch.Size([200, 64])
```

## ATK 测试

ATK 脚本目录：`experimental/atk_test/triton_add_position_embeddings_kernel/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/atk_test/triton_add_position_embeddings_kernel/npu_atk_test
```

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path add_position_embeddings_api.py
```

### 3.1 生成测试数据

```bash
atk case -f add_position_embeddings.yaml -p generate_add_position_embeddings.py
```

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_add_position_embeddings.json --task accuracy -p add_position_embeddings_api.py
```

注：如需保存精度对比数据则需要在指令后加 --save_data profile

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_add_position_embeddings.json --task performance -p add_position_embeddings_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile
