# _fbgemm_grouped_gemm_ws

## 来源

`_fbgemm_grouped_gemm_ws` 源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py>

当前套件测试 BF16 grouped GEMM 的 warp specialization（WS）路径。

## 算子功能

算子按 `m_sizes` 将 `x` 的 M 行切分为 G 组，并让每组输入与对应权重相乘：

```text
start_i = sum(m_sizes[:i])
x_i = x[start_i : start_i + m_sizes[i]]
w_i = w[i * N : (i + 1) * N]
y = concat(x_i @ w_i.T for i in range(G), dim=0)
```

WS 将 TMA 搬运与矩阵计算分配到不同 warp group。设备不支持 WS 或 TMA 时，算子自动回退到基础 grouped GEMM，数学语义保持不变。

## 输入与输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `(M, K)` | bf16 / fp16 / fp32 | 按行切分的输入矩阵 |
| `w` | `(G * N, K)` | 与 `x` 一致 | G 组堆叠的权重矩阵 |
| `m_sizes` | `(G,)` | int32 | 各组行数，总和为 M |

输出 `y` 的形状为 `(M, N)`，dtype 为 `bf16`。ATK API 会根据 M 和 G 确定性重建 `m_sizes` 的值。

## 关键约束

- 输入均为连续张量，`x`、`w` 必须是二维且 dtype 相同。
- `sum(m_sizes) == M`，`w.shape == (G * N, K)`，且 `K == w.shape[1]`。
- K 和 N 对齐到 8 的倍数；生成器保证 `M >= 64 * G`。
- API 固定 `use_fast_accum=True`、`_use_warp_specialization=True`。
- Ascend NPU、ROCm、非 Hopper GPU 或不满足 TMA 条件时回退到基础 kernel。
- 当前套件不覆盖 fused scatter-add；fp64 标杆输入在执行前转换为 fp32。

## 使用样例

```python
import torch
from _fbgemm_grouped_gemm_ws_kernel import grouped_gemm

M, K, N, G = 1024, 128, 256, 4
x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
w = torch.randn((G * N, K), dtype=x.dtype, device="cuda")
m_sizes = torch.tensor([256] * G, dtype=torch.int32, device="cuda")
y = grouped_gemm(x, w, m_sizes, _use_warp_specialization=True)
print(y.shape, y.dtype)
```

## 目录文件

```text
_fbgemm_grouped_gemm_ws/
├── README.md
├── gpu_atk_test/
│   ├── _fbgemm_grouped_gemm_ws_kernel.py
│   └── _fbgemm_grouped_gemm_ws_api.py
└── npu_atk_test/
    ├── _fbgemm_grouped_gemm_ws_kernel.py
    ├── _fbgemm_grouped_gemm_ws_api.py
    ├── _fbgemm_grouped_gemm_ws.yaml
    ├── generate_fbgemm_grouped_gemm_ws.py
    └── _fbgemm_grouped_gemm_ws_performance.json
```

## ATK 测试

NPU 端根据 YAML 和生成器生成用例，GPU 端启动 ATK server。两端用例 shape 完全一致，用于比较 GPU WS 路径与 NPU 回退路径。

```bash
atk case -f _fbgemm_grouped_gemm_ws.yaml -p generate_fbgemm_grouped_gemm_ws.py
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0     task -c all__fbgemm_grouped_gemm_ws.json --task accuracy -p _fbgemm_grouped_gemm_ws_api.py
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0     task -c _fbgemm_grouped_gemm_ws_performance.json --task performance_device     -p _fbgemm_grouped_gemm_ws_api.py
```

需要保存数据时，在命令末尾增加 `--save_data profile`。
