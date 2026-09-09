# triton_customized_gate_fwd_cublas_nhead

## 功能说明

triton_customized_gate_fwd_cublas_nhead 是一个基于 Triton 实现的 Ascend NPU 自定义算子，对应 CUDA 算子 `customized_gate_fwd_cublas_nhead` 的迁移，实现三阶段融合计算：

- Kernel 1（torch.addmm）：Gate GEMM，`gate_out = input @ gate_layer_weight^T + gate_layer_bias` → `(B, E)`
- Kernel 2（torch.baddbmm）：Proj GEMM，`proj_all = heads @ proj_weights^T + proj_biases` → `(H, B, Din)`
- Kernel 3（Triton fused kernel）：`clamp(-10, 10)` → SiLU → 按 head 求和 → 与 `input` 门控相乘 → RMSNorm → `(B, Din)`

### 输入

- `input` `(B, Din)`
- `gate_layer_weight` `(E, Din)`
- `gate_layer_bias` `(E,)`
- `proj_weights` `(H, Din, head_size)` — API 直接透传，与 CUDA 接口一致，无需 permute
- `proj_biases` `(H, Din)`
- `rms_norm_weight` `(Din,)`
- `eps`（标量，默认 `1e-5`）

### 输出

- `output` `(B, Din)`，即 `RMSNorm(Σ_h SiLU(clamp(proj_h)) * input)`

### 约束

- `E = H * head_size`（embedding_dim = num_heads * head_size）
- `proj_weights` 第 1 维必须等于 `Din`，`proj_biases` 形状必须为 `(H, Din)`
- 数据类型：bf16 / fp16（输出 dtype 与 `input` 一致）

## 目录结构

```text
experimental/triton/
├── atk_test/
│   └── triton_customized_gate_fwd_cublas_nhead/
│       ├── gpu_atk_test/
│       │   └── customized_gate_fwd_cublas_nhead_api.py      # GPU 端 ATK API 执行文件
│       └── npu_atk_test/
│           ├── customized_gate_fwd_cublas_nhead.yaml        # ATK 用例设计（dtype/shape/约束）
│           ├── generate_customized_gate_fwd_cublas_nhead.py # ATK 用例生成器
│           ├── customized_gate_fwd_cublas_nhead_api.py      # NPU 端 ATK API 执行文件
│           └── all_customized_gate_fwd_cublas_nhead.json    # ATK 生成的用例配置
└── customized_ops/
    └── triton_customized_gate_fwd_cublas_nhead/
        ├── README.md
        ├── triton_customized_gate_fwd_cublas_nhead.py   # Triton kernel 实现与对外 Python API
        └── test/
            └── test_customized_gate_fwd_cublas_nhead.py # pytest：Triton NPU vs PyTorch 精度对比
```

## 调用示例

通过 Python 直接导入调用（与 pytest 中用法一致）：

```python
import torch

from ops_rec.experimental.triton.triton_customized_gate_fwd_cublas_nhead import customized_gate_fwd_cublas_nhead

# case：his_gate（in_size=256, heads=4, embedding_dim=2048）
B, Din, H, Dh = 1000, 256, 4, 512
E = H * Dh

input = torch.randn(B, Din, dtype=torch.bfloat16, device="npu:2")
gate_layer_weight = torch.randn(E, Din, dtype=torch.bfloat16, device="npu:2")
gate_layer_bias = torch.randn(E, dtype=torch.bfloat16, device="npu:2")
proj_weights = torch.randn(H, Din, Dh, dtype=torch.bfloat16, device="npu:2")
proj_biases = torch.randn(H, Din, dtype=torch.bfloat16, device="npu:2")
rms_norm_weight = torch.randn(Din, dtype=torch.bfloat16, device="npu:2")

output = customized_gate_fwd_cublas_nhead(
    input, gate_layer_weight, gate_layer_bias,
    proj_weights, proj_biases, rms_norm_weight,
    eps=1e-5,
)
print(output.shape)  # torch.Size([1000, 256])
```

完整的精度对比用例（Triton NPU vs PyTorch 标杆）参见 [test_customized_gate_fwd_cublas_nhead.py](./test/test_customized_gate_fwd_cublas_nhead.py)。

## ATK 测试

ATK 脚本目录：`experimental/triton/atk_test/triton_customized_gate_fwd_cublas_nhead/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/triton/atk_test/triton_customized_gate_fwd_cublas_nhead/npu_atk_test
```

### 环境变量

```bash
export PYTHONPATH="experimental/triton/customized_ops/triton_customized_gate_fwd_cublas_nhead:${PYTHONPATH:-}"
```

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path customized_gate_fwd_cublas_nhead_api.py
```

### 3.1 生成测试数据

```bash
atk case -f customized_gate_fwd_cublas_nhead.yaml -p generate_customized_gate_fwd_cublas_nhead.py
```

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_customized_gate_fwd_cublas_nhead.json --task accuracy -p customized_gate_fwd_cublas_nhead_api.py
```

注：如需保存精度对比数据则需要在指令后加 --save_data profile

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_customized_gate_fwd_cublas_nhead.json --task performance -p customized_gate_fwd_cublas_nhead_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile
