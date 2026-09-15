# triton_jagged_to_dense

## 来源

`triton_jagged_to_dense` 算子源自 FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/c363b1b1077680d198235233f34af253f880d1fe/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py#L269>

## 算子功能

`triton_jagged_to_dense` 用于将 **jagged tensor（不规则张量）** 转换为 **dense tensor（稠密张量）**，并支持在写回时与一个 dense tensor 做元素级算术融合（`add` 或 `mul`）

- 通过 `jagged_max_lengths` 控制每一维的目标尺寸，不足部分以 `padding_value` 填充；越界部分直接截断。
- 可选地将 `jagged` 与同形状的 `operation_dense` 做元素级融合，结果直接写回 `output_dense`，无需中间临时张量。

---

## 输入

`triton_jagged_to_dense` 算子接受以下输入：

| 参数 | 说明 |
|------|------|
| `jagged_values` | jagged tensor 扁平值，shape `(N_total, inner_dim)`，`dtype` 支持 fp32/fp16/bf16 |
| `jagged_offsets` | 最内层分段偏移，要求首元素为 `0`、末元素为 `N_total` |
| `output_dense` | 输出 dense 张量，由调用方按目标形状预创建并提供指针 |
| `dense_indices` | 多维 jagged 时使用的 dense 索引；1D / 2D jagged 传任意张量即可 |
| `operation_function` | 融合操作，`"add"` / `"mul"` / `None` |
| `operation_dense` | 形状与 `output_dense` 一致，提供融合操作的右操作数；不融合时传 `None` |
| `JAGGED_DIM` | jagged tensor 的维度（`constexpr`），1D / 2D 时无需 `dense_indices` |
| `thread_block_row_size` | 块行大小（`constexpr`，常用 `32`） |
| `thread_block_col_size` | 块列大小（`constexpr`，常用 `32`） |

## 输出

`output_dense`：形状 `(B,) + tuple(jagged_max_lengths) + (inner_dim,)`，其中 `B = jagged_offsets[0].size(0) - 1`，`dtype` 与 `jagged_values` 相同。不足部分填充为 `padding_value`。

---

## 使用样例

### 示例 ：直接调用 kernel

```python
import torch
import torch_npu
from triton_jagged_to_dense import triton_jagged_to_dense

DEVICE = "npu:0"
values = torch.arange(6 * 2, dtype=torch.float32, device=DEVICE).reshape(6, 2)
jagged_offsets = torch.tensor([0, 1, 3, 4, 6], dtype=torch.int64, device=DEVICE)
output_dense = torch.full((2, 3, 2), -99.0, dtype=torch.float32, device=DEVICE)
dense_indices = torch.tensor([0, 2, 6, 8], dtype=torch.int32, device=DEVICE)

grid = (jagged_offsets.size(0) - 1,)
triton_jagged_to_dense[grid](
    values,
    jagged_offsets,
    values.stride(0),
    output_dense,
    dense_indices,
    output_dense.stride(-1),       # dense_col_stride
    output_dense.stride(-2),       # dense_row_stride
    output_dense.stride(-3),       # dense_matrix_stride
    JAGGED_DIM=3,                  # 2 层 offsets -> len(offsets) + 1 = 3
    operation_function=None,
    operation_dense=None,
)

torch.npu.synchronize(DEVICE)
kernel_out = output_dense.cpu()

# output_dense: [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]]
print("output_dense:", kernel_out.tolist())

```

---

## 关键约束

- `jagged_offsets[-1][0] == 0` 且 `jagged_offsets[-1][-1] == N_total`。
- 相邻两层 offsets 满足：`jagged_offsets[i][-1] == jagged_offsets[i+1].size(0) - 1`。
- `jagged_offsets[i]` 单调递增。

---

## 目录文件

```text
triton_jagged_to_dense/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_to_dense.py                # 原版算子 kernel 实现
│   └── triton_jagged_to_dense_api.py            # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── triton_jagged_to_dense.py                # NPU 适配优化后算子 kernel 实现
    ├── triton_jagged_to_dense_api.py            # NPU 端 ATK 自定义 API 封装
    ├── triton_jagged_to_dense.yaml              # ATK 算子用例参数
    ├── generate_triton_jagged_to_dense.py       # 限制参数生成脚本
    └── triton_jagged_to_dense_performance.json  # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/triton_jagged_to_dense_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成精度测试数据

```bash
atk case -f triton_jagged_to_dense.yaml -p generate_triton_jagged_to_dense.py
```

生成的精度测试用例位于 `result/triton_jagged_to_dense/json/` 下，文件名为 `all_triton_jagged_to_dense.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c all_triton_jagged_to_dense.json --task accuracy \
    -p triton_jagged_to_dense_api.py
```

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c triton_jagged_to_dense_performance.json --task performance_device \
    -p triton_jagged_to_dense_api.py
```
