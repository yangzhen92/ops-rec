# triton_batched_dense_vec_jagged_2d_matmul

## 来源

`triton_batched_dense_vec_jagged_2d_matmul` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py>

## 功能说明

triton_batched_dense_vec_jagged_2d_matmul 是 batched dense vector 与 jagged 2D tensor 的矩阵乘法，基于 Triton 实现。逻辑上等价于将 dense `[B*H, N_max]` 视为 `[B*H, 1, N_max]`、jagged values `[total_L, H*D]` 视为 `[B*H, N, D]`，逐行做 `(1, N) x (N, D)` 向量矩阵乘：

- GPU(CUDA) 端使用 FBGEMM 原生 kernel：grid 按 `(输出行, D 分块)` 划分，`grid = (B*H*ceil(D/32),)`，每个 program 计算输出的一行分块，K 方向逐行串行累加
- NPU(Ascend) 端使用迁移后的 persistent kernel：grid 固定为硬件向量核数（num_vectorcore），K 方向按 `BLOCK_K` 分块（autotune），每个 program 轮流处理所有 `(batch, head)` tile，纯 Vector Core 实现（无 tl.dot）

维度语义：

- `B = offset.size(0) - 1`（batch 数）
- `H = dense.size(0) // B`（head 数）
- `D = values.size(-1) // H`（embedding 维度）
- `N_max = dense.size(1)`（最大序列长度）
- `total_L = values.size(0) = offset[-1]`（values 总行数）

### 输入

- `dense` `(B*H, N_max)` — dense 向量
- `values` `(total_L, H*D)` — jagged 2D tensor 的 values
- `offset` `(B+1,)` — jagged offset，单调递增，`offset[0]=0`，`offset[-1]=total_L`

### 输出

- `output` `(B*H, D)` — dense 输出

### 约束

- `dense.shape[0] = B*H`，`dense.shape[1] = N_max`（第二维是序列长度，不是 D）
- `values.shape[0] = total_L = offset[-1]`，`values.shape[1] = H*D`
- `offset` 单调递增：`offset[0]=0`，`offset[-1]=total_L`
- 每段实际长度取 `min(end - begin, N_max)`，超长段被截断（kernel 内取 min）
- NPU 端 API 内统一将输入转为 fp32，避免 fp16/bf16 导致 kernel 内类型不一致
- NPU 端 offset 若为 int64 会预先转为 int32（NPU 不支持 int64，避免性能退化）

## 目录结构

```text
experimental/triton/atk_test/triton_batched_dense_vec_jagged_2d_matmul/
├── README.md
├── gpu_atk_test/
│   ├── triton_batched_dense_vec_jagged_2d_matmul.py       # GPU(CUDA) 端 Triton kernel（FBGEMM 原生实现）
│   └── triton_batched_dense_vec_jagged_2d_matmul_api.py   # GPU(CUDA) 端 ATK API 执行文件
└── npu_atk_test/
    ├── triton_batched_dense_vec_jagged_2d_matmul.py           # NPU(Triton) kernel 实现（persistent grid + autotune）
    ├── triton_batched_dense_vec_jagged_2d_matmul.yaml         # ATK 用例设计（dtype/shape/约束）
    ├── generate_triton_batched_dense_vec_jagged_2d_matmul.py  # ATK 用例生成器（维度约束链）
    ├── triton_batched_dense_vec_jagged_2d_matmul_api.py       # NPU(Triton) 端 ATK API 执行文件
    └── all_triton_batched_dense_vec_jagged_2d_matmul.json     # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用：

```python
import torch

from triton_batched_dense_vec_jagged_2d_matmul import batched_dense_vec_jagged_2d_matmul

# case：B=4 个 batch，H=8 个 head，D=64，N_max=32
B, H, D, N_max = 4, 8, 64, 32
total_L = 128

dense = torch.randn(B * H, N_max, dtype=torch.float32, device="npu:2")
values = torch.randn(total_L, H * D, dtype=torch.float32, device="npu:2")
offset = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device="npu:2")

output = batched_dense_vec_jagged_2d_matmul(dense, values, offset)
print(output.shape)  # torch.Size([32, 64])
```

## ATK 测试

ATK 脚本目录：`experimental/triton/atk_test/triton_batched_dense_vec_jagged_2d_matmul/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/triton/atk_test/triton_batched_dense_vec_jagged_2d_matmul/npu_atk_test
```

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path triton_batched_dense_vec_jagged_2d_matmul_api.py
```

### 3.1 生成测试数据

```bash
atk case -f triton_batched_dense_vec_jagged_2d_matmul.yaml -p generate_triton_batched_dense_vec_jagged_2d_matmul.py
```

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all_triton_batched_dense_vec_jagged_2d_matmul.json --task accuracy -p triton_batched_dense_vec_jagged_2d_matmul_api.py
```

注：如需保存精度对比数据则需要在指令后加 --save_data profile

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all_triton_batched_dense_vec_jagged_2d_matmul.json --task performance -p triton_batched_dense_vec_jagged_2d_matmul_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile
