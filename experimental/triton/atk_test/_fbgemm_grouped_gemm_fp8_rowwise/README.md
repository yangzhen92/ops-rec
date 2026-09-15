# _fbgemm_grouped_gemm_fp8_rowwise

## 来源

`_fbgemm_grouped_gemm_fp8_rowwise` 算子源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/5d0a3d9868e04a0e0dd30906c79a96a001b85f68/fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py#L615>

当前 ATK 套件覆盖 FP8 行级量化的基础 grouped GEMM 路径。NPU 端使用包含
Ascend 配置剪枝的 autotune kernel，GPU 端使用适配 99 KB 共享内存上限的固定配置。

## 算子功能

算子先把高精度输入 `x` 和权重 `w` 按行量化为 FP8，并生成与量化结果严格耦合的
倒数尺度。随后按 `m_sizes` 将 `x` 的 M 行切分为 G 组，让每组输入与对应权重相乘：

```text
(x_fp8, x_scale) = quantize_fp8_row(x)
(w_fp8, w_scale) = quantize_fp8_row(w)
start_i = sum(m_sizes[j] for j in range(i))
y_i = (x_fp8[start_i : start_i + m_sizes[i]] @ w_fp8[i * N : (i + 1) * N].T)
      * x_scale[start_i : start_i + m_sizes[i], None]
      * w_scale[i * N : (i + 1) * N, None].T
y = concat(y_0, y_1, ..., y_(G-1), dim=0)
```

矩阵乘使用 FP32 累加，输出转换为 BF16。各组共享 K 和 N，但可以有不同的行数。

---

## 输入

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `(M, K)` | bf16 / fp16 / fp32 | API 按行量化为 FP8 的输入矩阵 |
| `w` | `(G * N, K)` | 与 `x` 一致 | G 组堆叠的权重矩阵，API 按行量化为 FP8 |
| `m_sizes` | `(G,)` | int32 | 各组输入行数，要求总和为 M |

`x_scale` 和 `w_scale` 不由 YAML 独立生成。API 在量化 `x`、`w` 时同步生成这两个
倒数尺度，保证 FP8 张量与尺度在数值上匹配。

ATK 只能生成 `m_sizes` 的形状，无法注入总和恰好为 M 的确定序列。因此 API 根据
M 和 G 均分行数，并将余数依次分配给前面的组。

## 输出

`y`：形状为 `(M, N)`，dtype 为 `bf16`。第 i 组输出形状为
`(m_sizes[i], N)`，各组按输入顺序拼接。

---

## 关键约束

- `x`、`w` 必须是二维连续张量，量化前 dtype 保持一致。
- `m_sizes` 必须是一维连续 int32 张量，且 `G = m_sizes.numel() > 0`。
- `sum(m_sizes) == M`，`w.shape == (G * N, K)`。
- `K == w.shape[1]`，`w.shape[0]` 必须能被 G 整除。
- FP8 路径要求 K 和 N 都是 8 的倍数；生成器会向下对齐并保证最小值为 8。
- 生成器保证 `M >= 64 * G`，避免 autotune 配置被全部剪枝。
- API 固定 `use_fast_accum=True` 和 `_use_warp_specialization=False`。
- 当前套件不覆盖 fused scatter-add 路径。
- ROCm 优先使用 `float8_e4m3fnuz`，其他平台使用 `float8_e4m3fn`。

---

## 使用样例

```python
import torch

from _fbgemm_grouped_gemm_fp8_rowwise_api import _quantize_fp8_row
from _fbgemm_grouped_gemm_fp8_rowwise_kernel import grouped_gemm_fp8_rowwise

device = "npu"
M, K, N, G = 1024, 128, 256, 4
x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
w = torch.randn((G * N, K), dtype=x.dtype, device=device)
m_sizes = torch.tensor([256, 256, 256, 256], dtype=torch.int32, device=device)
fp8_dtype = torch.float8_e4m3fn
x_fp8, x_scale = _quantize_fp8_row(torch, x, fp8_dtype)
w_fp8, w_scale = _quantize_fp8_row(torch, w, fp8_dtype)

y = grouped_gemm_fp8_rowwise(
    x_fp8,
    w_fp8,
    m_sizes,
    x_scale,
    w_scale,
    use_fast_accum=True,
    _use_warp_specialization=False,
)
print(y.shape, y.dtype)  # torch.Size([1024, 256]) torch.bfloat16
```

---

## 目录文件

```text
_fbgemm_grouped_gemm_fp8_rowwise/
├── README.md
├── gpu_atk_test/
│   ├── _fbgemm_grouped_gemm_fp8_rowwise_kernel.py
│   └── _fbgemm_grouped_gemm_fp8_rowwise_api.py
└── npu_atk_test/
    ├── _fbgemm_grouped_gemm_fp8_rowwise_kernel.py
    ├── _fbgemm_grouped_gemm_fp8_rowwise_api.py
    ├── _fbgemm_grouped_gemm_fp8_rowwise.yaml
    ├── generate_fbgemm_grouped_gemm_fp8_rowwise.py
    └── _fbgemm_grouped_gemm_fp8_rowwise_performance.json
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端，`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端根据 YAML 和生成器生成。
- 性能测试使用 `_fbgemm_grouped_gemm_fp8_rowwise_performance.json` 中的 150 个用例。
- GPU 端启动 ATK server，NPU 端作为客户端发起精度和性能测试。

### 生成精度测试数据

在 NPU 端的 `npu_atk_test/` 目录执行：

```bash
atk case -f _fbgemm_grouped_gemm_fp8_rowwise.yaml \
    -p generate_fbgemm_grouped_gemm_fp8_rowwise.py
```

生成文件位于 `result/_fbgemm_grouped_gemm_fp8_rowwise/json/`，文件名为
`all__fbgemm_grouped_gemm_fp8_rowwise.json`。

### 远端 GPU 服务器启动监听

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c _fbgemm_grouped_gemm_fp8_rowwise_performance.json --task accuracy \
    -p _fbgemm_grouped_gemm_fp8_rowwise_api.py
```

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c _fbgemm_grouped_gemm_fp8_rowwise_performance.json --task performance_device \
    -p _fbgemm_grouped_gemm_fp8_rowwise_api.py
```

如需保存精度或性能数据，在对应命令末尾增加 `--save_data profile`。
