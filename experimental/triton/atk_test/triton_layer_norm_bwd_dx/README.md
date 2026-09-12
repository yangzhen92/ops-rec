# triton_layer_norm_bwd_dx

## 来源

`triton_layer_norm_bwd_dx` 算子源自 NVIDIA recsys-examples 代码仓中的 Triton layer normalization kernel 实现：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_layer_norm.py>

## 算子功能

`triton_layer_norm_bwd_dx` 用于计算非 learnable layer normalization 反向传播过程中输入张量 `X` 的梯度 `DX`。

- 输入 `X` 和上游梯度 `DY` 均为二维张量，shape 为 `[N, D]`。
- 每个 Triton program 处理一行数据，执行网格为 `grid = (N,)`。
- ATK API 根据 `X` 按行推导前向计算所需的 `Mean` 和 `Rstd`，保证 GPU / NPU 两端输入一致。
- kernel 内部使用 fp32 完成归一化和归约计算，结果转换为 `DX` 的 dtype 后写回。
- NPU 版本使用 `@triton.autotune` 根据特征维 `D` 自动选择 `BLOCK_D`。

对每一行的计算公式如下：

```text
xhat = (X - Mean) * Rstd
c1 = sum(xhat * DY) / D
c2 = sum(DY) / D
DX = (DY - (xhat * c1 + c2)) * Rstd
```

---

## 输入

| 参数 | 说明 |
|------|------|
| `DX` | 输入梯度输出张量，shape 为 `[N, D]`，dtype 与 `X` 一致 |
| `DY` | 上游梯度，shape 为 `[N, D]`，支持 `fp32` / `fp16` / `bf16` |
| `X` | layer normalization 前向输入，shape 为 `[N, D]`，支持 `fp32` / `fp16` / `bf16` |
| `Mean` | `X` 的逐行均值，shape 为 `[N]`，dtype 为 `fp32` |
| `Rstd` | `1 / sqrt(var + eps)` 的逐行结果，shape 为 `[N]`，dtype 为 `fp32` |
| `stride_dx` | `DX` 的行 stride |
| `stride_dy` | `DY` 的行 stride |
| `stride_x` | `X` 的行 stride |
| `D` | 特征维度，即 `X.shape[1]` |
| `eps` | layer normalization 的数值稳定常数；ATK API 使用它推导 `Rstd` |
| `BLOCK_D` | `constexpr`，每个 program 覆盖的特征维长度 |

在 ATK 用例中只需提供 `X`、`DY` 和 `eps`，`Mean`、`Rstd` 及 `DX` 均由 API 封装创建。

## 输出

`DX`：输入张量 `X` 的梯度，shape 为 `[N, D]`，dtype 与 `X` 一致。

---

## 使用示例

### 示例：直接调用 NPU autotune kernel

```python
import torch

from triton_layer_norm_bwd_dx_kernel import _layer_norm_bwd_dx

x = torch.randn((64, 1024), dtype=torch.float16, device="npu")
dy = torch.randn_like(x)
dx = torch.empty_like(x)

eps = 1e-5
x_fp32 = x.float()
mean = x_fp32.mean(dim=1)
var = x_fp32.var(dim=1, unbiased=False)
rstd = torch.rsqrt(var + eps)

_layer_norm_bwd_dx[(x.shape[0],)](
    dx,
    dy,
    x,
    mean,
    rstd,
    dx.stride(0),
    dy.stride(0),
    x.stride(0),
    x.shape[1],
    eps,
)
```

---

## 关键约束

- `X` 和 `DY` 必须是非空二维张量，且 shape 均为 `[N, D]`。
- `DX` 必须与 `X` shape 一致，dtype 与 `X` 一致。
- `Mean` 和 `Rstd` 必须是 shape 为 `[N]` 的 fp32 张量。
- `BLOCK_D` 必须满足 `BLOCK_D >= next_power_of_2(D)`，否则无法覆盖完整特征维。
- `BLOCK_D` 受单个 program 的 64KB 元素访问规模限制，需满足 `BLOCK_D <= 65536 // X.element_size()`。
- 当前 ATK 配置将 `D` 限制在 `1` 至 `8192`，并限制 `N * D` 以避免测试数据占用过多内存。

---

## 目录文件

```text
triton_layer_norm_bwd_dx/
├── README.md
├── gpu_atk_test/
│   ├── triton_layer_norm_bwd_dx_kernel.py           # GPU 侧原始 kernel 实现
│   └── triton_layer_norm_bwd_dx_api.py              # GPU 侧 ATK 自定义 API 封装
└── npu_atk_test/
    ├── triton_layer_norm_bwd_dx_kernel.py           # NPU 适配后的 autotune kernel 实现
    ├── triton_layer_norm_bwd_dx_api.py              # NPU 侧 ATK 自定义 API 封装
    ├── triton_layer_norm_bwd_dx.yaml                # ATK 算子用例参数
    ├── generate_triton_layer_norm_bwd_dx.py         # shape 约束生成脚本
    └── triton_layer_norm_bwd_dx_performance.json    # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 侧；`gpu_atk_test/` 部署到 GPU 侧。
- 精度测试用例在 NPU 侧生成。
- 性能测试用例复用 `npu_atk_test/triton_layer_norm_bwd_dx_performance.json`。
- GPU 侧启动 ATK server，NPU 侧作为 ATK 客户端拉起精度 / 性能测试。

### 生成精度测试数据

该命令读取 YAML 参数空间，并通过 generator 修正 X/DY 的 shape 关系，生成精度测试用例 JSON。

```bash
atk case -f triton_layer_norm_bwd_dx.yaml -p generate_triton_layer_norm_bwd_dx.py
```

生成的精度测试用例位于 `result/triton_layer_norm_bwd_dx/json/` 下，文件名为 `all_triton_layer_norm_bwd_dx.json`。

### 远端 GPU 服务器启动监听命令

该命令在 GPU 基准环境启动 ATK server，并从指定目录加载 GPU API 封装。

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

该命令由 NPU 侧连接 GPU server，分别计算两端的 DX，并比较输出梯度的精度。

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_triton_layer_norm_bwd_dx.json --task accuracy \
    -p triton_layer_norm_bwd_dx_api.py
```

### 性能测试

该命令读取性能用例 JSON，在 NPU 侧执行设备性能测试并记录 backward kernel 耗时。

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c triton_layer_norm_bwd_dx_performance.json --task performance_device \
    -p triton_layer_norm_bwd_dx_api.py
```
