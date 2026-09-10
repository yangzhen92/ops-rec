# _fbgemm_grouped_gemm

## 来源

`_fbgemm_grouped_gemm` 源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py>

当前套件测试基础 grouped GEMM，关闭 warp specialization、FP8 和 fused scatter-add。

## 算子功能

算子按 `m_sizes` 将 `x` 的 M 行切分为 G 组，并让每组输入与对应权重相乘：

```text
start_i = sum(m_sizes[:i])
x_i = x[start_i : start_i + m_sizes[i]]
w_i = w[i * N : (i + 1) * N]
y = concat(x_i @ w_i.T for i in range(G), dim=0)
```

各组共享 K 和 N，可以具有不同的行数，输出保持原始分组顺序。

## 输入与输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `(M, K)` | bf16 / fp16 / fp32 | 按行切分的输入矩阵 |
| `w` | `(G * N, K)` | 与 `x` 一致 | G 组堆叠的权重矩阵 |
| `m_sizes` | `(G,)` | int32 | 各组行数，总和为 M |

输出 `y` 的形状为 `(M, N)`，dtype 固定为 `bf16`。ATK 无法直接生成满足总和约束的 `m_sizes`，API 会根据 M 和 G 确定性重建其值。

## 关键约束

- 输入均为连续张量，`x`、`w` 必须是二维且 dtype 相同。
- `m_sizes` 是一维 int32 张量，`G = m_sizes.numel() > 0`。
- `sum(m_sizes) == M`，`w.shape == (G * N, K)`，且 `K == w.shape[1]`。
- K 和 N 向下对齐到 8 的倍数，最小值为 8。
- 生成器保证 `M >= 64 * G`，避免 autotune 配置被全部剪枝。
- API 固定 `use_fast_accum=True`、`_use_warp_specialization=False`。
- 当前套件不覆盖 fused scatter-add；fp64 标杆输入在执行前转换为 fp32。

## 使用样例

```python
import torch
from _fbgemm_grouped_gemm_kernel import grouped_gemm

M, K, N, G = 1024, 128, 256, 4
x = torch.randn((M, K), dtype=torch.bfloat16, device="npu")
w = torch.randn((G * N, K), dtype=x.dtype, device="npu")
m_sizes = torch.tensor([256] * G, dtype=torch.int32, device="npu")
y = grouped_gemm(x, w, m_sizes, _use_warp_specialization=False)
print(y.shape, y.dtype)
```

## 目录文件

```text
_fbgemm_grouped_gemm/
├── README.md
├── gpu_atk_test/
│   ├── _fbgemm_grouped_gemm_kernel.py
│   └── _fbgemm_grouped_gemm_api.py
└── npu_atk_test/
    ├── _fbgemm_grouped_gemm_kernel.py
    ├── _fbgemm_grouped_gemm_api.py
    ├── _fbgemm_grouped_gemm.yaml
    ├── generate_fbgemm_grouped_gemm.py
    └── _fbgemm_grouped_gemm_performance.json
```

## ATK 测试

`npu_atk_test/` 部署到 NPU 端，`gpu_atk_test/` 部署到 GPU 端。NPU 端根据 YAML 和生成器生成精度用例，性能测试使用 JSON 中的 150 个用例。

在 NPU 端生成精度用例：

```bash
atk case -f _fbgemm_grouped_gemm.yaml -p generate_fbgemm_grouped_gemm.py
```

在 GPU 端启动 ATK server：

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

在 NPU 端运行精度测试：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0     task -c all__fbgemm_grouped_gemm.json --task accuracy -p _fbgemm_grouped_gemm_api.py
```

运行性能测试：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0     task -c _fbgemm_grouped_gemm_performance.json --task performance_device     -p _fbgemm_grouped_gemm_api.py
```

需要保存数据时，在命令末尾增加 `--save_data profile`。
