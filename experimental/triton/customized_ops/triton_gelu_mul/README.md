# triton_gelu_mul

## 功能说明

triton_gelu_mul 是一个基于 Triton 实现的 Ascend NPU 自定义算子：

- 输入 `input` 形状为 `[..., 2d]`，将最后一维从中间切分为两半：
  - `x1 = input[..., :d]`
  - `x2 = input[..., d:]`
- 输出 `out = GELU(x1) * x2`，形状为 `[..., d]`。
- 支持两种 GELU 实现，通过 `approximate` 参数选择：
  - `"none"`（默认）：精确 erf 实现

### 约束

- `input` 维度数：2 ~ 8
- 最后一维必须为偶数（2d）
- 数据类型：fp32 / fp16 / bf16

## 目录结构

```
experimental/triton/
├── atk_test/
│   └── triton_gelu_mul/
│       └── npu_atk_test/
│           ├── triton_gelu_mul.yaml        # ATK 用例设计（dtype/shape/属性约束）
│           ├── generate_triton_gelu_mul.py # ATK 用例生成器（保证最后一维为偶数）
│           ├── triton_gelu_mul_api.py      # ATK API 执行文件（triton/npu/cpu 多后端注册）
│           └── all_triton_gelu_mul.json    # ATK 生成的用例配置
└── customized_ops/
    └── triton_gelu_mul/
        ├── README.md
        ├── triton_gelu_mul.py              # Triton kernel 实现与对外 Python API
        └── test/
            └── test_gelu_mul.py            # pytest：Triton NPU vs PyTorch CPU 精度对比
```

## 调用示例

通过 Python 直接导入调用（与 pytest 中用法一致）：

```python
import torch

from ops_rec.experimental.triton.customized_ops.triton_gelu_mul import triton_gelu_mul

# 输入 [..., 2d]：最后一维必须为偶数，前一半作为 x1，后一半作为 x2
input_npu = torch.randn(1000, 3072, dtype=torch.bfloat16, device="npu:2")

# approximate: "none"（精确 erf）或 "tanh"（近似）
output = triton_gelu_mul(input_npu, approximate="none")
print(output.shape)  # torch.Size([1000, 1536])
```

完整的精度对比用例（Triton NPU vs PyTorch CPU 标杆）参见 [test_gelu_mul.py](./test/test_gelu_mul.py)。

## ATK 测试

ATK 脚本目录：`experimental/triton/atk_test/triton_gelu_mul/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/triton/atk_test/triton_gelu_mul/npu_atk_test
```

### 环境变量

```bash
export PYTHONPATH="experimental/triton/customized_ops/triton_gelu_mul:${PYTHONPATH:-}"
```

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path triton_gelu_mul_api.py
```

### 3.1 生成测试数据

```bash
atk case -f triton_gelu_mul.yaml -p generate_triton_gelu_mul.py
```

### 3.2 精度对比 — 双标杆 Triton vs CPU

```bash
atk node --backend triton --devices 0 node --backend cpu task -c all_triton_gelu_mul.json --task accuracy -p triton_gelu_mul_api.py
```

注：如需保存精度对比数据则需要在指令后加 --save_data profile 

### 3.3 性能对比 — Triton vs AscendC(npu)

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 task -c all_triton_gelu_mul.json --task performance_device -p triton_gelu_mul_api.py
```
注：如需保存性能对比数据则需要在指令后加 --save_data profile 
