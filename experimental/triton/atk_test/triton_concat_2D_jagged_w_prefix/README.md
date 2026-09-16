# triton_concat_2D_jagged_w_prefix

## 来源

`concat_2D_jagged_w_prefix` 算子源自 Meta generative-recommenders：

<https://github.com/meta-recsys/generative-recommenders/blob/4a33ba4405ebf85599acae40527e6ffb964292f7/generative_recommenders/ops/triton/triton_jagged.py>

## 功能说明

该算子按 batch 将两段二维 jagged tensor 拼接到同一个输出中。当前 ATK 用例覆盖
`IS_DENSE_A=False`、`IS_DENSE_B=False`、`IS_REPLACE=False`、`n_prefix_from_B=0` 的
jagged + jagged 拼接路径：

```text
Out[OffsetsA[b] + OffsetsB[b] + i, :] =
    ValuesA[OffsetsA[b] + i, :]  if i < len_a[b]
    ValuesB[OffsetsB[b] + i - len_a[b], :] otherwise
```

该 kernel 是纯数据搬运算子，不做数值规约。NPU 实现与 GPU 的分支选择方式保持一致：
每个 program 先按当前 token 选择 A 或 B 的输入行基址，再沿 `D` 维分块完成 load/store。
`@triton.autotune` 以 `D` 为 key，在 `BLOCK_D=32/64/128/256/512/1024/2048` 中选择配置；
API 不传入 `BLOCK_D`，由 autotune 注入。

### 输入

- `seq_offsets` `(B + 1,)`：ATK 用例中仅使用其长度确定 batch size。
- `ValuesA` `(L_A, D)`：第一段 jagged 数据。
- `ValuesB` `(L_B, D)`：第二段 jagged 数据。

### 输出

- `Out` `(L_A + L_B, D)`：按 batch 拼接后的输出。

### 约束

- `ValuesA` 和 `ValuesB` 必须是二维 tensor，且第二维 `D` 相同。
- `ValuesA` 和 `ValuesB` 的 dtype 必须一致，支持 `fp32`、`fp16`、`bf16`。
- `L_A >= B` 且 `L_B >= B`，保证每个 batch 至少一个 token。
- API 会根据 shape 确定性重建 `OffsetsA/OffsetsB`，传入的随机 offset 数值不参与 kernel。
- `max_seq_len * B` 不能超过 NPU coreDim 限制。

## 目录结构

```text
experimental/triton/atk_test/triton_concat_2D_jagged_w_prefix/
├── README.md
├── gpu_atk_test/
│   ├── triton_concat_2D_jagged_w_prefix_kernel.py
│   └── triton_concat_2D_jagged_w_prefix_api.py
└── npu_atk_test/
    ├── triton_concat_2D_jagged_w_prefix_kernel.py
    ├── triton_concat_2D_jagged_w_prefix_api.py
    ├── triton_concat_2D_jagged_w_prefix.yaml
    ├── generate_triton_concat_2D_jagged_w_prefix.py
    └── triton_concat_2D_jagged_w_prefix_performance.json
```

## 调用示例

```python
import torch

from triton_concat_2D_jagged_w_prefix_kernel import concat_2D_jagged_w_prefix

device = "npu"
b, d = 4, 64
offsets_a = torch.tensor([0, 2, 5, 7, 8], device=device, dtype=torch.int64)
offsets_b = torch.tensor([0, 1, 3, 4, 6], device=device, dtype=torch.int64)
values_a = torch.randn((8, d), device=device, dtype=torch.float16)
values_b = torch.randn((6, d), device=device, dtype=torch.float16)
out = torch.empty((14, d), device=device, dtype=values_a.dtype)
max_seq_len = 5

concat_2D_jagged_w_prefix[(max_seq_len, b)](
    offsets_a, values_a, offsets_b, values_b, 0, out, d,
    values_a.stride(0), values_b.stride(0), 0, out.stride(0), 0,
    IS_DENSE_A=False, IS_DENSE_B=False, IS_REPLACE=False,
)
```

## ATK 测试

### 1. NPU 端生成精度用例

该命令根据 YAML 采样输入，并由 generator 对齐 `B/L_A/L_B/D`：

```bash
atk case -f triton_concat_2D_jagged_w_prefix.yaml -p generate_triton_concat_2D_jagged_w_prefix.py
```

生成的精度用例文件名为 `all_triton_concat_2D_jagged_w_prefix.json`。

### 2. GPU 端启动 ATK 服务

该命令在 GPU 端加载 `gpu_atk_test/` 中的 API 和 kernel：

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3. NPU 端执行精度对比

该命令从 NPU 端连接 GPU 服务，比较两端拼接输出：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_concat_2D_jagged_w_prefix.json --task accuracy -p triton_concat_2D_jagged_w_prefix_api.py
```

### 4. NPU 端执行性能对比

该命令使用固定性能用例比较两端设备耗时：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_concat_2D_jagged_w_prefix_performance.json --task performance_device -p triton_concat_2D_jagged_w_prefix_api.py
```
