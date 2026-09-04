# triton_add_position_embeddings_bwd

## 来源

`add_position_embeddings_bwd` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 功能说明

triton_add_position_embeddings_bwd 是 add_position_embeddings 前向算子的反向 kernel，基于 Triton 实现，用于计算位置编码（Dense 查找表）的梯度累加：

- 前向中每行 `n` 通过 clamp 索引 `idx = min(n, high_inds[b])` 从 Dense 表取值，反向时该行的梯度累加到 `DenseOut[idx]`
- 对每个 batch `b`（行区间 `[seq_offsets[b], seq_offsets[b+1])`）：
  - 行 `n < high_inds[b]`：`DenseOut[n] += Jagged[n]`
  - 行 `n >= high_inds[b]`：全部累加到 `DenseOut[high_inds[b]]`
- `SCALE_JAGGED`（scale != 1.0）时，同时对参与累加的行写回 `JaggedOut[n] = Jagged[n] * scale`（API 中 JaggedOut 复用 Jagged）

### 输入

- `Jagged` `(N, D)` — 前向输入的 jagged tensor
- `seq_offsets` `(B+1,)` — 每个 batch 的起止行偏移
- `high_inds` `(B,)` — 每个 batch 的 dense 表最大 lookup 索引
- `DenseOut` `(K, D)` — 预分配的输出（kernel 写入结果）
- `scale`（标量，默认 `1.0`）— 缩放系数

### 输出

- `DenseOut` `(max(high_inds)+1, D)` — 各 batch 中 clamp 到该索引的 Jagged 行之和

### 约束

- `Jagged.shape[0] = seq_offsets[-1]`（总行数 = 最后一个 offset）
- `seq_offsets` 单调递增，首 0 尾 N
- `high_inds[i] < seq_offsets[i+1] - seq_offsets[i]`（每段高位索引不超过段长）
- `DenseOut.shape[1] = Jagged.shape[1]`（D 维度对齐），dtype 一致
- 数据类型：fp32 / fp16 / bf16

## 目录结构

```text
experimental/triton/atk_test/triton_add_position_embeddings_bwd/
├── README.md
├── gpu_atk_test/
│   ├── triton_add_position_embeddings_bwd_kernel.py   # GPU(CUDA) 端 Triton kernel 实现
│   └── triton_add_position_embeddings_bwd_api.py      # GPU(CUDA) 端 ATK API 执行文件
└── npu_atk_test/
    ├── triton_add_position_embeddings_bwd_kernel.py   # NPU(Triton) kernel 实现
    ├── add_position_embeddings.yaml                   # ATK 用例设计（dtype/shape/约束）
    ├── generate_add_position_embeddings_bwd.py        # ATK 用例生成器（保证 B/D 维度与 dtype 一致）
    ├── add_position_embeddings_bwd_api.py             # NPU(Triton) 端 ATK API 执行文件
    └── all_add_position_embeddings.json               # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中用法一致）：

```python
import torch

from triton_add_position_embeddings_bwd_kernel import _add_position_embeddings_bwd_kernel

# case：B=2 个 batch，D=64，high_inds=[30, 60]
jagged = torch.randn(200, 64, dtype=torch.bfloat16, device="npu:2")
seq_offsets = torch.tensor([0, 100, 200], dtype=torch.int32, device="npu:2")
high_inds = torch.tensor([30, 60], dtype=torch.int32, device="npu:2")

max_high_ind = int(high_inds.max().item())  # 60
dense_out = torch.zeros(max_high_ind + 1, 64, dtype=jagged.dtype, device="npu:2")

ncores, rows_per_core = 56, (max_high_ind + 1 + 56 - 1) // 56
_add_position_embeddings_bwd_kernel[(ncores,)](
    jagged, seq_offsets, high_inds, dense_out, jagged,
    2, 64, 1.0, rows_per_core,
    jagged.stride(0), jagged.stride(0), dense_out.stride(0),
    SCALE_JAGGED=False, BLOCK_D=64, BLOCK_K=rows_per_core,
)
print(dense_out.shape)  # torch.Size([61, 64])
```

## ATK 测试

ATK 脚本目录：`experimental/triton/atk_test/triton_add_position_embeddings_bwd/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/triton/atk_test/triton_add_position_embeddings_bwd/npu_atk_test
```

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path triton_add_position_embeddings_bwd_api.py
```

### 3.1 生成测试数据

```bash
atk case -f add_position_embeddings.yaml -p generate_add_position_embeddings_bwd.py
```

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all_add_position_embeddings.json --task accuracy -p add_position_embeddings_bwd_api.py
```

注：如需保存精度对比数据则需要在指令后加 --save_data profile

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all_add_position_embeddings.json --task performance -p add_position_embeddings_bwd_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile
