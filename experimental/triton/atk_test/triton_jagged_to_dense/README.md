# triton_jagged_to_dense

## 来源

`triton_jagged_to_dense` 算子源自 FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/v1.5.0-release/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py>

## 算子功能

`triton_jagged_to_dense` 用于将 **jagged tensor（不规则张量）** 转换为 **dense tensor（稠密张量）**，并支持在写回时与一个 dense tensor 做元素级算术融合（`add` 或 `mul`）。

- 支持 **1D / 2D / N-D jagged tensor**。
- 通过 `jagged_max_lengths` 控制每一维的目标尺寸，不足部分以 `padding_value` 填充；越界部分直接截断。
- 可选地将 `jagged` 与同形状的 `operation_dense` 做元素级融合，结果直接写回 `output_dense`，无需中间临时张量。
- 多维 jagged 时通过 `_jagged_offsets_to_dense_indice` 计算每个 jagged 段在 dense 中的索引。

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
import triton
from triton_jagged_to_dense import triton_jagged_to_dense

values = torch.tensor(
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
    dtype=torch.float32, device="npu",
)
inner_offsets = torch.tensor([0, 3, 7], dtype=torch.int64, device="npu")
output_dense = torch.zeros(2, 4, 2, dtype=torch.float32, device="npu")
dense_indices = torch.zeros(2, dtype=torch.int32, device="npu")

grid = (inner_offsets.size(0) - 1,)
triton_jagged_to_dense[grid](
    values,
    inner_offsets,
    values.stride(0),
    output_dense,
    dense_indices,
    output_dense.stride(-1),       # dense_col_stride
    output_dense.stride(-2),       # dense_row_stride
    output_dense.stride(-3),       # dense_matrix_stride
    JAGGED_DIM=2,
    thread_block_row_size=32,
    thread_block_col_size=32,
    operation_function=None,
    operation_dense=None,
)
```

---

## 关键约束

- `jagged_offsets[-1][0] == 0` 且 `jagged_offsets[-1][-1] == N_total`。
- 相邻两层 offsets 满足：`jagged_offsets[i][-1] == jagged_offsets[i+1].size(0) - 1`。
- 当 `JAGGED_DIM > 2` 时，dense 越界段（`dense_indice == -1`）会被跳过，输出保留 `padding_value`。

---

## 目录文件

| 文件 | 说明 |
|------|------|
| `triton_jagged_to_dense.py` | `triton_jagged_to_dense` 算子 kernel 实现 |
| `atk/triton_jagged_to_dense_api.py` | ATK 自定义 API 封装 |
| `atk/jagged_to_dense.yaml` | ATK 算子用例参数 |
| `atk/generate_jagged_to_dense.py` | 限制参数生成脚本 |
| `atk/triton_jagged_to_dense_performce.json` | 性能测试样例数据 |

---

## 单精度测试

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_triton_jagged_to_dense.json --task accuracy \
    -p triton_jagged_to_dense_api.py
```

### 性能测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c triton_jagged_to_dense_performce.json --task accuracy \
    -p triton_jagged_to_dense_api.py
```