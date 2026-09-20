# _weighted_rms_norm_fwd

## 来源

`triton_weighted_rms_norm_fwd` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/a247bd4baf4dc74c3e19b4f2ed1fd1b62eb75586/examples/hstu/ops/triton_ops/triton_layer_norm.py>

## 功能说明

该算子对二维输入 `X` 的每一行执行带权 RMS Normalization。它不减去均值，先计算
均方根的倒数，再将归一化结果与逐特征权重 `W` 相乘：

```text
Rstd[n] = 1 / sqrt(sum(X[n, d]^2) / D + eps)
Y[n, d] = X[n, d] * Rstd[n] * W[d]
```

kernel 内部使用 fp32 计算平方和、`Rstd` 及权重乘法，最终将 `Y` 转回输入 dtype。

### 输入

- `X`：形状为 `[N, D]` 的二维输入张量，支持 `fp32`、`fp16` 和 `bf16`。
- `W`：形状为 `[D]` 的一维权重张量，dtype 与 `X` 对齐。
- `Y`：ATK 输入中的输出占位张量，API wrapper 会重新分配实际输出。
- `Rstd`：ATK 输入中的输出占位张量，API wrapper 会重新分配 fp32 输出。
- `eps`：防止除零的浮点标量，测试范围为 `[1e-6, 1e-4]`。

### 输出

- `Y`：形状为 `[N, D]`，dtype 与 `X` 相同。
- `Rstd`：形状为 `[N]`，dtype 为 `fp32`。

### 实现说明

grid 为 `(N,)`，每个 Triton program 负责一行。`BLOCK_D` 覆盖完整特征维，超出
`D` 的位置通过 mask 屏蔽。GPU wrapper 根据 `D` 和元素大小直接计算 `BLOCK_D` 与
`num_warps`；NPU kernel 使用 `D` 作为 autotune key，从候选 `BLOCK_D` 中剪除无法覆盖
整行或超过 64 KB 元素容量限制的配置。

## 目录结构

```text
experimental/triton/atk_test/weighted_rms_norm_fwd/
├── README.md
├── gpu_atk_test/
│   ├── triton_weighted_rms_norm_fwd_kernel.py
│   └── triton_weighted_rms_norm_fwd_api.py
└── npu_atk_test/
    ├── triton_weighted_rms_norm_fwd_kernel.py
    ├── triton_weighted_rms_norm_fwd_api.py
    ├── triton_weighted_rms_norm_fwd.yaml
    ├── generate_triton_weighted_rms_norm_fwd.py
    └── triton_weighted_rms_norm_fwd_performance.json
```

- `*_kernel.py`：分别提供 GPU 固定配置和 NPU autotune kernel。
- `*_api.py`：解析 ATK 输入、创建输出张量并启动对应 kernel。
- `*.yaml`：定义精度用例的 dtype、shape、数值范围及边界条件。
- `generate_*.py`：让 `Y` 与 `X` 同形、`W` 为 `[D]`、`Rstd` 为 `[N]`，并对齐 dtype。
- `*_performance.json`：保存固定的性能测试用例。

## 调用示例

```python
import torch

from triton_weighted_rms_norm_fwd_kernel import _weighted_rms_norm_fwd

n, d = 128, 1024
x = torch.randn((n, d), device="npu", dtype=torch.float16)
w = torch.randn((d,), device="npu", dtype=x.dtype)
y = torch.empty_like(x)
rstd = torch.empty((n,), device="npu", dtype=torch.float32)

_weighted_rms_norm_fwd[(n,)](
    x,
    y,
    w,
    rstd,
    d,
    1e-5,
    x.stride(0),
    y.stride(0),
)
```

## ATK 测试

测试由 GPU server 与 NPU client 协同完成。GPU 端部署 `gpu_atk_test/`，NPU 端部署
`npu_atk_test/`。

### 1. NPU 端生成精度用例

以下命令根据 YAML 配置采样输入，并通过 generator 对齐关联张量的 shape 和 dtype：

```bash
atk case -f triton_weighted_rms_norm_fwd.yaml -p generate_triton_weighted_rms_norm_fwd.py
```

命令生成精度任务文件 `all_triton_weighted_rms_norm_fwd.json`。

### 2. GPU 端启动 ATK 服务

以下命令在 GPU 服务器上加载 `gpu_atk_test/` 中的 API 文件并监听任务请求：

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3. NPU 端执行精度对比

以下命令连接 GPU 服务，在 NPU 与 GPU 上运行相同用例并比较 `Y` 和 `Rstd`：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_weighted_rms_norm_fwd.json --task accuracy -p triton_weighted_rms_norm_fwd_api.py
```

如需保留精度对比数据，可在命令末尾添加 `--save_data profile`。

### 4. NPU 端执行性能对比

以下命令使用固定性能用例，比较 NPU 与 GPU 的设备执行时间：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_weighted_rms_norm_fwd_performance.json --task performance_device -p triton_weighted_rms_norm_fwd_api.py
```

如需保留性能数据，可在命令末尾添加 `--save_data profile`。
