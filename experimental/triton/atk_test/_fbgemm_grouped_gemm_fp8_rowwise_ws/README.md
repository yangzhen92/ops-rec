# _fbgemm_grouped_gemm_fp8_rowwise_ws

## 来源

`_fbgemm_grouped_gemm_fp8_rowwise_ws` 算子源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/5d0a3d9868e04a0e0dd30906c79a96a001b85f68/fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py#L791>

该变体在 FP8 行级量化 grouped GEMM 上请求 Warp Specialization（WS）。支持 WS 与 TMA
的 NVIDIA GPU 执行 WS Kernel；NPU 不支持该路径，运行时给出提示并回退到相同数值语义的
FP8 Rowwise 基础 Kernel，用于与 GPU 结果逐用例比较。

## 算子功能

输入 `x` 和 `w` 先按行量化为 FP8，并同步生成倒数尺度。`m_sizes` 将 `x` 的 M 行切成
G 组，第 i 组与 `w` 中对应的 `(N, K)` 权重相乘：

```text
(x_fp8, x_scale) = quantize_fp8_row(x)
(w_fp8, w_scale) = quantize_fp8_row(w)
y_i = (x_fp8_i @ w_fp8_i.T) * x_scale_i[:, None] * w_scale_i[None, :]
y = concat(y_0, ..., y_(G-1), dim=0)
```

矩阵乘使用 FP32 累加，输出为 BF16。

## 输入与输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `(M, K)` | bf16 / fp16 / fp32 | 按行量化的输入矩阵 |
| `w` | `(G * N, K)` | 与 `x` 一致 | G 组堆叠的权重矩阵 |
| `m_sizes` | `(G,)` | int32 | 每组行数，API 会重建并保证总和为 M |
| `y` | `(M, N)` | bf16 | 各组输出按顺序拼接 |

`x_scale` 与 `w_scale` 由 API 量化时生成，避免随机尺度与 FP8 数据不匹配。

## 关键约束

- `x`、`w` 是二维连续张量，且 K 相同。
- `m_sizes` 是一维连续 int32 张量，`G > 0` 且 `sum(m_sizes) == M`。
- `w.shape == (G * N, K)`，K 和 N 均为 8 的倍数。
- 生成器保证 `M >= 64 * G`，并将 K、N 向下对齐到 8 的倍数。
- API 固定 `use_fast_accum=True` 和 `_use_warp_specialization=True`。
- fused scatter-add 路径不在当前套件范围内。

## 使用样例

```python
import torch

from _fbgemm_grouped_gemm_fp8_rowwise_ws_api import _quantize_fp8_row
from _fbgemm_grouped_gemm_fp8_rowwise_ws_kernel import grouped_gemm_fp8_rowwise

device = "npu"
M, K, N, G = 1024, 128, 256, 4
x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
w = torch.randn((G * N, K), dtype=x.dtype, device=device)
m_sizes = torch.tensor([256] * G, dtype=torch.int32, device=device)
x_fp8, x_scale = _quantize_fp8_row(torch, x, torch.float8_e4m3fn)
w_fp8, w_scale = _quantize_fp8_row(torch, w, torch.float8_e4m3fn)
y = grouped_gemm_fp8_rowwise(
    x_fp8, w_fp8, m_sizes, x_scale, w_scale,
    use_fast_accum=True, _use_warp_specialization=True,
)
print(y.shape, y.dtype)
```

## 目录文件

```text
_fbgemm_grouped_gemm_fp8_rowwise_ws/
├── README.md
├── gpu_atk_test/
│   ├── _fbgemm_grouped_gemm_fp8_rowwise_ws_api.py
│   └── _fbgemm_grouped_gemm_fp8_rowwise_ws_kernel.py
└── npu_atk_test/
    ├── _fbgemm_grouped_gemm_fp8_rowwise_ws.yaml
    ├── _fbgemm_grouped_gemm_fp8_rowwise_ws_api.py
    ├── _fbgemm_grouped_gemm_fp8_rowwise_ws_kernel.py
    ├── _fbgemm_grouped_gemm_fp8_rowwise_ws_performance.json
    └── generate_fbgemm_grouped_gemm_fp8_rowwise_ws.py
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3.1 生成测试数据

```bash
atk case -f _fbgemm_grouped_gemm_fp8_rowwise_ws.yaml \
    -p generate_fbgemm_grouped_gemm_fp8_rowwise_ws.py
```

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c _fbgemm_grouped_gemm_fp8_rowwise_ws_performance.json --task accuracy \
    -p _fbgemm_grouped_gemm_fp8_rowwise_ws_api.py
```

注：如需保存精度对比数据，需要在命令后加 `--save_data profile`。

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c _fbgemm_grouped_gemm_fp8_rowwise_ws_performance.json --task performance_device \
    -p _fbgemm_grouped_gemm_fp8_rowwise_ws_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
