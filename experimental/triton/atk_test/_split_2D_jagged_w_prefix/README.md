# _split_2D_jagged_w_prefix

## 来源

`split_2D_jagged_w_prefix` 算子源自 Meta generative-recommenders：

<https://github.com/meta-recsys/generative-recommenders/blob/4a33ba4405ebf85599acae40527e6ffb964292f7/generative_recommenders/ops/triton/triton_jagged.py>

## 功能说明

该算子是 `concat_2D_jagged_w_prefix` 的反向数据搬运形式：输入 `JaggedIn[L, D]`
中按 batch 连续存放 A、B 两段 jagged 数据，kernel 根据 `OffsetsA/OffsetsB` 将其拆回
`OutA[L_A, D]` 和 `OutB[L_B, D]`。当前 ATK 用例覆盖
`IS_DENSE_A=False`、`IS_DENSE_B=False`、`IS_REPLACE=False`、`n_prefix_to_B=0` 的路径。

```text
JaggedIn[OffsetsA[b] + OffsetsB[b] + i, :] ->
    OutA[OffsetsA[b] + i, :]  if i < len_a[b]
    OutB[OffsetsB[b] + i - len_a[b], :] otherwise
```

NPU 实现与 GPU 的分支选择方式保持一致：每个 program 先按当前 token 选择 `OutA` 或
`OutB` 的输出行基址，再沿 `D` 维分块完成 load/store。`@triton.autotune` 以 `D` 为 key，
在 `BLOCK_D=32/64/128/256/512/1024/2048` 中选择配置；API 不传入 `BLOCK_D`，由 autotune 注入。

### 输入

- `JaggedIn` `(L, D)`：拼接后的 jagged 输入。
- `OffsetsA` `(B + 1,)`：ATK 用例中仅使用其长度确定 batch size。
- `OffsetsB` `(B + 1,)`：长度必须与 `OffsetsA` 一致。

### 输出

- `OutA` `(L_A, D)`：拆分后的 A 段。
- `OutB` `(L_B, D)`：拆分后的 B 段。

### 约束

- `JaggedIn` 必须是二维 tensor，支持 `fp32`、`fp16`、`bf16`。
- `OffsetsA` 与 `OffsetsB` 必须是一维 `int64` tensor，长度相同且至少为 2。
- API 会根据 shape 确定性重建合法 offsets，传入的随机 offset 数值不参与 kernel。
- `L >= B`，并且 generator 会控制 `max_seq_len * B` 不超过 NPU coreDim 限制。

## 目录结构

```text
experimental/triton/atk_test/_split_2D_jagged_w_prefix/
├── README.md
├── gpu_atk_test/
│   ├── triton_split_2D_jagged_w_prefix_kernel.py
│   └── triton_split_2D_jagged_w_prefix_api.py
└── npu_atk_test/
    ├── triton_split_2D_jagged_w_prefix_kernel.py
    ├── triton_split_2D_jagged_w_prefix_api.py
    ├── triton_split_2D_jagged_w_prefix.yaml
    ├── generate_triton_split_2D_jagged_w_prefix.py
    └── triton_split_2D_jagged_w_prefix_performance.json
```

## 调用示例

```python
import torch

from triton_split_2D_jagged_w_prefix_kernel import split_2D_jagged_w_prefix

device = "npu"
b, d = 4, 64
offsets_a = torch.tensor([0, 2, 5, 7, 8], device=device, dtype=torch.int64)
offsets_b = torch.tensor([0, 1, 3, 4, 6], device=device, dtype=torch.int64)
jagged = torch.randn((14, d), device=device, dtype=torch.float16)
out_a = torch.empty((8, d), device=device, dtype=jagged.dtype)
out_b = torch.empty((6, d), device=device, dtype=jagged.dtype)
max_seq_len = 5

split_2D_jagged_w_prefix[(max_seq_len, b)](
    jagged, 0, offsets_a, offsets_b, out_a, out_b, d,
    jagged.stride(0), out_a.stride(0), out_b.stride(0), 0,
    IS_DENSE_A=False, IS_DENSE_B=False, IS_REPLACE=False,
)
```

## ATK 测试

### 1. NPU 端生成精度用例

该命令根据 YAML 采样输入，并由 generator 对齐 `B/L/D`：

```bash
atk case -f triton_split_2D_jagged_w_prefix.yaml -p generate_triton_split_2D_jagged_w_prefix.py
```

生成的精度用例文件名为 `all_triton_split_2D_jagged_w_prefix.json`。

### 2. GPU 端启动 ATK 服务

该命令在 GPU 端加载 `gpu_atk_test/` 中的 API 和 kernel：

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3. NPU 端执行精度对比

该命令从 NPU 端连接 GPU 服务，比较两端拆分输出：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_split_2D_jagged_w_prefix.json --task accuracy -p triton_split_2D_jagged_w_prefix_api.py
```

### 4. NPU 端执行性能对比

该命令使用固定性能用例比较两端设备耗时：

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_split_2D_jagged_w_prefix_performance.json --task performance_device -p triton_split_2D_jagged_w_prefix_api.py
```
