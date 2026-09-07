# triton_layer_norm_fwd

## 来源

`triton_layer_norm_fwd` 算子源自 Meta Platforms 的 Triton layer normalization kernel 实现。本目录保留原始版权声明，并在 NPU/ATK 适配文件中补充 Huawei Technologies Co., Ltd. 版权与 Apache License 声明。

## 算子功能

`triton_layer_norm_fwd` 用于对二维输入张量 `X [N, D]` 的每一行执行 layer normalization 前向计算。

- 每个 Triton program 处理输入中的一行，`grid = (N,)`。
- 在 kernel 内按行计算 `mean` 和 `rstd`，并将归一化结果写入 `Y`。
- `Y` 与 `X` 保持相同 shape 和 dtype。
- `Mean` 与 `Rstd` 为一维 fp32 输出，shape 为 `[N]`。
- NPU 版本使用 `@triton.autotune` 自动选择 `BLOCK_D`，并按 `D` 与 dtype 对候选配置剪枝。

---

## 输入

| 参数 | 说明 |
|------|------|
| `X` | 输入张量，shape 为 `[N, D]`，支持 `fp32` / `fp16` / `bf16` |
| `Y` | 输出张量，shape 为 `[N, D]`，dtype 与 `X` 一致 |
| `Mean` | 输出每行均值，shape 为 `[N]`，dtype 为 `fp32` |
| `Rstd` | 输出每行 `1 / sqrt(var + eps)`，shape 为 `[N]`，dtype 为 `fp32` |
| `D` | 特征维度，即 `X.shape[1]` |
| `eps` | layer normalization epsilon |
| `stride_x` | `X` 的行 stride |
| `stride_y` | `Y` 的行 stride |
| `TRAINING` | `constexpr`，为 `True` 时写出 `Mean` / `Rstd` |
| `BLOCK_D` | `constexpr`，每个 program 覆盖的特征维长度 |
| `COMPUTE_MEAN_AND_RSTD` | `constexpr`，为 `True` 时在 kernel 内计算 `mean` / `rstd` |

## 输出

- `Y`：归一化后的输出张量，shape 为 `[N, D]`，dtype 与 `X` 一致。
- `Mean`：每行均值，shape 为 `[N]`，dtype 为 `fp32`。
- `Rstd`：每行标准差倒数，shape 为 `[N]`，dtype 为 `fp32`。

---

## 使用示例

### 示例：直接调用 kernel

```python
import torch
import triton
from triton_layer_norm import _layer_norm_fwd

x = torch.randn((64, 1024), dtype=torch.float16, device="npu")
y = torch.empty_like(x)
mean = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
rstd = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)

d = x.shape[1]
block_d = triton.next_power_of_2(d)
num_warps = min(max(block_d // 256, 1), 8)
grid = (x.shape[0],)

_layer_norm_fwd[grid](
    x,
    y,
    mean,
    rstd,
    d,
    1e-5,
    x.stride(0),
    y.stride(0),
    TRAINING=True,
    BLOCK_D=block_d,
    COMPUTE_MEAN_AND_RSTD=True,
    num_warps=num_warps,
)
```

---

## 关键约束

- `X` 必须是二维张量，shape 为 `[N, D]`，且 `N > 0`。
- `Y` 必须与 `X` shape 一致，dtype 与 `X` 一致。
- `Mean` 与 `Rstd` 必须为 shape `[N]` 的 fp32 张量。
- `BLOCK_D` 必须覆盖整行特征维，即 `BLOCK_D >= next_power_of_2(D)`。
- `BLOCK_D` 受单 program 64KB 元素访问规模限制，通常满足 `BLOCK_D <= 65536 // X.element_size()`。

---

## 目录文件

```text
triton_lay_norm_fwd/
├── README.md
├── gpu_atk_test/
│   ├── triton_layer_norm.py                  # GPU 侧原始 kernel 实现
│   └── triton_layer_norm_fwd_api.py          # GPU 侧 ATK 自定义 API 封装
└── npu_atk_test/
    ├── triton_layer_norm.py                  # NPU 适配后的 autotune kernel 实现
    ├── triton_layer_norm_fwd_api.py          # NPU 侧 ATK 自定义 API 封装
    ├── triton_layer_norm_fwd.yaml            # ATK 算子用例参数
    ├── generate_triton_layer_norm_fwd.py     # shape / dtype 约束生成脚本
    └── triton_layer_norm_fwd_performance.json # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 侧；`gpu_atk_test/` 部署到 GPU 侧。
- 精度测试用例在 NPU 侧生成。
- 性能测试用例复用 `npu_atk_test/triton_layer_norm_fwd_performance.json`。
- GPU 侧启动 ATK server，NPU 侧作为 ATK 客户端拉起精度 / 性能测试。

### 生成精度测试数据

```bash
atk case -f triton_layer_norm_fwd.yaml -p generate_triton_layer_norm_fwd.py
```

生成的精度测试用例位于 `result/triton_layer_norm_fwd/json/` 下，文件名为 `all_triton_layer_norm_fwd.json`。

### 远端 GPU 服务器启动监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_triton_layer_norm_fwd.json --task accuracy \
    -p triton_layer_norm_fwd_api.py
```

### 性能测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c triton_layer_norm_fwd_performance.json --task performance_device \
    -p triton_layer_norm_fwd_api.py
```
