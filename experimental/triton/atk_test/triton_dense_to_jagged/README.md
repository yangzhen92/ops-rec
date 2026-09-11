# triton_dense_to_jagged

## 来源

`triton_dense_to_jagged` 算子源自 FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/v1.5.0-release/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py>

## 算子功能

`triton_dense_to_jagged` 用于将 **dense tensor（稠密张量）** 转换为 **jagged tensor（不规则张量）**，并支持在写出时与一个 jagged tensor 做元素级算术融合（`add` 或 `mul`）。

- 通过 `jagged_offsets` 描述 dense 中哪些切片被抽取、抽取后的 jagged tensor 形状如何。
- 对于多维 jagged（`JAGGED_DIM > 2`），通过 `dense_indices` 把 `pid` 映射到 dense 张量中的偏移；若对应位置越界，`dense_indices` 标记为 `-1`，kernel 跳过该 row。
- 可选地将 dense 与同形状的 `operation_jagged_values` 做元素级融合，结果直接写回 `output_jagged_values`，无需中间临时张量。

---

## 输入

`triton_dense_to_jagged` 算子接受以下输入：

| 参数 | 说明 |
|------|------|
| `jagged_value_ptr` | 输出 jagged tensor 的扁平值指针，shape `(N_total, inner_dim)`，`dtype` 支持 fp32/fp16/bf16 |
| `jagged_offsets_ptr` | 最内层分段偏移，要求首元素为 `0`、末元素为 `N_total` |
| `jagged_value_row_stride` | 输出 jagged tensor 在行维度上的 stride（通常等于 `inner_dim`） |
| `output_dense_ptr` | 输入 dense 张量指针；`JAGGED_DIM == 2` 时按 `pid * dense_matrix_stride` 偏移，`JAGGED_DIM > 2` 时按 `dense_indices[pid]` 偏移 |
| `dense_indices_ptr` | 多维 jagged 时使用的 dense 索引；1D / 2D jagged 传任意张量即可，越界位置填 `-1` |
| `dense_col_stride` | dense 最后一维 stride |
| `dense_row_stride` | dense 倒数第二维 stride |
| `dense_matrix_stride` | dense 倒数第三维 stride |
| `JAGGED_DIM` | jagged tensor 的维度（`constexpr`），等于 `len(jagged_offsets) + 1`；1D / 2D 时无需 `dense_indices` |
| `thread_block_row_size` | 块行大小（`constexpr`，NPU 端会通过 `triton.autotune` 自动选择；GPU 端常用 `32`） |
| `thread_block_col_size` | 块列大小（`constexpr`，NPU 端会通过 `triton.autotune` 自动选择；GPU 端常用 `32`） |
| `operation_function` | 融合操作，`"add"` / `"mul"` / `None` |
| `operation_jagged_value_ptr` | 形状与 `output_jagged_value` 一致，提供融合操作的右操作数；不融合时传 `None` |

## 输出

`output_jagged_value`：形状 `(N_total, inner_dim)`，其中 `N_total = jagged_offsets[-1][-1]`，`inner_dim = dense.size(-1)`，`dtype` 与 `dense` 相同。对于越界位置（`dense_indices[pid] == -1`）的 row，kernel 不会从 dense 读取，但若 `operation_function` 不为 `None`，仍会进行融合计算（值会被 mask 填 0 后参与）。

---

## 使用样例

### 示例 ：直接调用 kernel

```python
import torch
import torch_npu
from triton_dense_to_jagged import triton_dense_to_jagged

DEVICE = "npu:0"
dense = torch.arange(1 * 5 * 3, dtype=torch.float32, device=DEVICE).reshape(1, 5, 3)
# jagged_offsets: shape [B+1]，每个 batch 抽取 seq 个元素（这里只取前 3 个，裁掉后 2 个）
jagged_offsets = torch.tensor([0, 3], dtype=torch.int64, device=DEVICE)
# 输出 jagged tensor 的 shape 为 (N_total, D) = (3, 3)
output_jagged_value = torch.empty((3, 3), dtype=torch.float32, device=DEVICE)
dense_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)

grid = (jagged_offsets.size(0) - 1,)
triton_dense_to_jagged[grid](
    output_jagged_value,                       # 输出：jagged tensor
    jagged_offsets,                            # 输入：offsets
    output_jagged_value.stride(0),             # 输出 jagged tensor 行 stride
    dense,                                     # 输入：dense tensor
    dense_indices,                             # dense indices
    dense.stride(-1),                          # dense_col_stride
    dense.stride(-2),                          # dense_row_stride
    dense.stride(-3),                          # dense_matrix_stride
    JAGGED_DIM=2,                              # 1 层 offsets -> len(offsets) + 1 = 2
    operation_function=None,
    operation_jagged_value_ptr=None,
)

torch.npu.synchronize(DEVICE)
kernel_out = output_jagged_value.cpu()
print(dense)
# output_jagged_value: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
print("output_jagged_value:", kernel_out.tolist())

```

---

## 关键约束

- `jagged_offsets[-1][0] == 0` 且 `jagged_offsets[-1][-1] == N_total`。
- `jagged_offsets[i]` 单调递增；相邻两层 offsets 满足 `jagged_offsets[i][-1] == jagged_offsets[i+1].size(0) - 1`。
- `JAGGED_DIM == 2` 时 dense 至少为 3D；`JAGGED_DIM > 2` 时 dense 至少为 4D，且 `dense_indices[pid]` 越界时填 `-1`。
- `operation_jagged_values` 的 `dtype` 与 `dense` 保持一致；当 `operation_function` 为 `None` 时 `operation_jagged_value_ptr` 传 `None`。
- dense 最后一维（特征维 `D`）建议 `<= 8192`，且内部维满足 `B < seq < D`（3D）或 `B1 < B2 < seq < D`（4D）的算子使用约束。

---

## 目录文件

```text
triton_dense_to_jagged/
├── README.md
├── gpu_atk_test/
│   ├── triton_dense_to_jagged.py                # 原版算子 kernel 实现
│   └── triton_dense_to_jagged_api.py            # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── triton_dense_to_jagged.py                # NPU 适配优化后算子 kernel 实现（带 autotune）
    ├── triton_dense_to_jagged_api.py            # NPU 端 ATK 自定义 API 封装
    ├── triton_dense_to_jagged.yaml              # ATK 算子用例参数
    ├── generate_triton_dense_to_jagged.py       # 限制参数生成脚本
    └── triton_dense_to_jagged_performance.json  # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/triton_dense_to_jagged_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成精度测试数据

```bash
atk case -f triton_dense_to_jagged.yaml -p generate_triton_dense_to_jagged.py
```

生成的精度测试用例位于 `result/triton_dense_to_jagged/json/` 下，文件名为 `all_triton_dense_to_jagged.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c all_triton_dense_to_jagged.json --task accuracy \
    -p triton_dense_to_jagged_api.py
```

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c triton_dense_to_jagged_performance.json --task performance_device \
    -p triton_dense_to_jagged_api.py
```
