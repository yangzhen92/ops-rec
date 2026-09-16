# triton_weighted_rms_norm_bwd_dx

## 来源

`_weighted_rms_norm_bwd_dx` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/a247bd4baf4dc74c3e19b4f2ed1fd1b62eb75586/examples/hstu/ops/triton_ops/triton_layer_norm.py>

## 功能说明

该算子计算 Weighted RMSNorm 反向传播中的输入梯度 `DX`，并生成分组的 partial
权重梯度 `DW`。对输入 `X[N, D]`、上游梯度 `DY[N, D]`、权重 `W[D]` 和前向阶段的
`Rstd[N]`，核心计算为：

```text
xhat[n, d] = X[n, d] * Rstd[n]
wdy[n, d] = W[d] * DY[n, d]
c1[n] = sum(xhat[n, :] * wdy[n, :]) / D
DX[n, d] = (wdy[n, d] - xhat[n, d] * c1[n]) * Rstd[n]
partial_DW[group, d] += DY[n, d] * xhat[n, d]
```

GPU 版本使用原始自旋锁规约 partial `DW`；NPU 版本使用初始化为 0 的 `DW` 并通过
`tl.atomic_add` 累加 partial weight gradient，避免 NPU 编译器不支持自旋锁结构。

### 输入

- `X` `(N, D)`：前向输入，支持 `fp32`、`fp16`、`bf16`。
- `DY` `(N, D)`：输出梯度，shape 与 `X` 一致。
- `W` `(D,)`：RMSNorm 权重，dtype 与 `X` 对齐。
- `Rstd` `(N,)`：前向阶段保存的标准差倒数，API 转为 `fp32`。
- `eps`：与原始 kernel 签名保持一致的 epsilon 标量。

### 输出

- `DX` `(N, D)`：反向输入梯度。
- `DW` `(GROUP_N, D)`：API 内部分配的 partial weight gradient 缓冲区，用于后续规约。

### 约束

- `X` 和 `DY` 必须是 shape 相同的二维 tensor。
- `W` 必须是一维 tensor，长度等于 `D`。
- `Rstd` 必须是一维 tensor，长度等于 `N`。
- NPU autotune 版本限制 `D <= 8192`，否则 `BLOCK_D=16384` 会超过 UB 容量。
- `X`、`DY`、`W` 的 dtype 对齐；`Rstd` 使用 `fp32`。

## 目录结构

```text
experimental/triton/atk_test/triton_weighted_rms_norm_bwd_dx/
├── README.md
├── gpu_atk_test/
│   ├── triton_weighted_rms_norm_bwd_dx_kernel.py
│   └── triton_weighted_rms_norm_bwd_dx_api.py
└── npu_atk_test/
    ├── triton_weighted_rms_norm_bwd_dx_kernel.py
    ├── triton_weighted_rms_norm_bwd_dx_api.py
    ├── triton_weighted_rms_norm_bwd_dx.yaml
    ├── generate_triton_weighted_rms_norm_bwd_dx.py
    └── triton_weighted_rms_norm_bwd_dx_performance.json
```

## 调用示例

```python
import torch
import triton

from triton_weighted_rms_norm_bwd_dx_kernel import _weighted_rms_norm_bwd_dx

n, d = 128, 1024
device = "npu"
x = torch.randn((n, d), device=device, dtype=torch.float16)
dy = torch.randn_like(x)
w = torch.randn((d,), device=device, dtype=x.dtype)
rstd = torch.rand((n,), device=device, dtype=torch.float32)
dx = torch.empty_like(x)
group_n = min(n, 2048)
dw = torch.zeros((group_n, d), device=device, dtype=torch.float32)
locks = torch.zeros((2 * group_n,), device=device, dtype=torch.int32)

_weighted_rms_norm_bwd_dx[(n,)](
    dx, dy, dw, x, w, rstd, locks,
    dx.stride(0), dy.stride(0), x.stride(0),
    d, 1e-5, GROUP_N=group_n,
)
```

## ATK 测试

### 1. NPU 端生成精度用例

该命令根据 YAML 随机采样输入，并由 generator 对齐 `X/DY/W/Rstd` 的 shape 与 dtype：

```bash
atk case -f triton_weighted_rms_norm_bwd_dx.yaml -p generate_triton_weighted_rms_norm_bwd_dx.py
```

生成的精度用例文件名为 `all_triton_weighted_rms_norm_bwd_dx.json`。

### 2. GPU 端启动 ATK 服务

该命令在 GPU 端加载 `gpu_atk_test/` 中的 API 和 kernel：

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3. NPU 端执行精度对比

该命令从 NPU 端连接 GPU 服务，分别运行两端实现并比较 `DX`：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_weighted_rms_norm_bwd_dx.json --task accuracy -p triton_weighted_rms_norm_bwd_dx_api.py
```

### 4. NPU 端执行性能对比

该命令使用仓库内固定性能用例比较两端设备耗时：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_weighted_rms_norm_bwd_dx_performance.json --task performance_device -p triton_weighted_rms_norm_bwd_dx_api.py
```
