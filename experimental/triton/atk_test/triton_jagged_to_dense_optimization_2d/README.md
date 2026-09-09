# triton_jagged_to_dense_optimization_2d

## 来源

`triton_jagged_to_dense_optimization_2d` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py>

## 功能说明

`triton_jagged_to_dense_optimization_2d` 将二维 jagged tensor（`jagged_values` + `jagged_offsets`）转换为固定行长的 dense tensor：每个 program 负责一个 sub-tensor，把变长行拷贝/填充到 `[outer, max_len, inner]` 的输出中，超出的位置写入 `padding_value`。可选传入 `operation_function` 做融合操作（本实现中调用时未启用，传 `None`）。

### 输入

- `jagged_values` `(total_elements, inner_size)` — 拼接存储的二维值
- `jagged_offsets` `(outer + 1,)` — 每行起止索引，单调递增，`offsets[0] = 0`，`offsets[-1] = total_elements`
- `jagged_max_lengths` `[max_len]` — dense 输出行数（2D 场景为单元素列表）
- `padding_value` — 越界位置的填充值，dtype 与 `jagged_values` 一致
- `operation_function` / `operation_value` — 融合操作配置（未启用）

### 输出

- `output_dense` `(outer, max_len, inner_size)` — 对应 dense 结果

### 约束

- `jagged_values` 必须是二维 contiguous tensor，支持 `fp32`、`fp16`、`bf16`
- `jagged_offsets` 支持 `int32`、`int64`，长度 = `outer + 1`
- NPU 端通过 `@triton.autotune` 自动选择 `thread_block_row_size` / `thread_block_col_size`；GPU 端固定 tile 大小

## 目录结构

```text
triton_jagged_to_dense_optimization_2d/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_to_dense_optimization_2d.py       # GPU 端 Triton kernel 实现
│   └── triton_jagged_to_dense_optimization_2d_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── triton_jagged_to_dense_optimization_2d.py       # NPU 端 Triton kernel 实现（autotune）
    ├── triton_jagged_to_dense_optimization_2d.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_triton_jagged_to_dense_optimization_2d.py  # ATK 用例生成器
    ├── triton_jagged_to_dense_optimization_2d_api.py   # NPU 端 ATK API 执行文件
    └── triton_jagged_to_dense_optimization_2d_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch

from triton_jagged_to_dense_optimization_2d import triton_jagged_to_dense_optimization_2d

inner_size, max_len, outer = 4, 2, 3
total_elements = outer * max_len
device = "npu"
values = torch.arange(total_elements * inner_size, dtype=torch.float32, device=device).reshape(total_elements, inner_size)
offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=device)
dense = torch.empty((outer, max_len, inner_size), dtype=values.dtype, device=device)

grid = (outer,)
triton_jagged_to_dense_optimization_2d[grid](
    values,
    offsets,
    values.stride(0),
    dense,
    dense.stride(-2),
    dense.stride(-3),
    thread_block_row_size=32,
    thread_block_col_size=32,
    padded_value=0.0,
    operation_function=None,
    operation_dense=None,
)
print(dense.shape)  # torch.Size([3, 2, 4])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/triton_jagged_to_dense_optimization_2d_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f triton_jagged_to_dense_optimization_2d.yaml -p generate_triton_jagged_to_dense_optimization_2d.py
```

生成器会把 `jagged_offsets` 的 shape 固定为 `[outer + 1]`，并把 `jagged_max_lengths` 收敛为单元素列表。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_triton_jagged_to_dense_optimization_2d.json --task accuracy \
    -p triton_jagged_to_dense_optimization_2d_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c triton_jagged_to_dense_optimization_2d_performance.json --task performance_device \
    -p triton_jagged_to_dense_optimization_2d_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
