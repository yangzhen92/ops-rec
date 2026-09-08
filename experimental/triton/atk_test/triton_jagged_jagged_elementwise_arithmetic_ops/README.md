# `triton_jagged_jagged_elementwise_arithmetic_ops` 算子文档

## 来源

`jagged_jagged_elementwise_arithmetic_ops` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py>

## 功能说明

`jagged_jagged_elementwise_arithmetic_ops` 基于 Triton 实现，用于对两个 jagged tensor 的 values 部分执行逐元素加法或乘法。两个输入均以二维矩阵 `(M, N)` 表示，计算公式为：

```text
output[m, n] = x_values[m, n] + y_values[m, n]  # ops_func == "add"
output[m, n] = x_values[m, n] * y_values[m, n]  # ops_func == "mul"
```

该 kernel 不接收 jagged offsets，只处理已经展平且逻辑位置相互对齐的 values 矩阵。因此，两个输入对应的原始 jagged tensor 应具有相同的 offsets 和特征维度，这一语义约束由调用方保证。

kernel 使用一维 program id 映射二维 tile：

```text
num_group_n = ceil_div(N, thread_block_col_size)
pid_n = pid % num_group_n
pid_m = pid // num_group_n
```

每个 program 处理一个 `(thread_block_row_size, thread_block_col_size)` 数据块，并通过 mask 保护 `M`、`N` 边界。

当前 GPU/NPU ATK API 均固定传入 `ops_func="add"`，所以现有 ATK 用例验证的是逐元素加法；底层 kernel 同时保留 `ops_func="mul"` 的乘法能力。

## 输入

| 参数 | 说明 |
|------|------|
| `x_values` | 第一个 jagged tensor 的展平 values，shape `(M, N)` |
| `y_values` | 第二个 jagged tensor 的展平 values，shape `(M, N)`，shape、dtype 和内存布局需与 `x_values` 一致 |

YAML 用例支持 `fp32`、`fp16` 和 `bf16`。用例生成器会将 `y_values` 的 shape 和 dtype 对齐到 `x_values`。

底层 kernel 还接收以下启动参数，这些参数由 ATK API 根据输入自动计算或设置：

| 参数 | 说明 |
|------|------|
| `M` | values 矩阵的行数，即 `x_values.shape[0]` |
| `N` | values 矩阵的列数，即 `x_values.shape[1]` |
| `stride_row` | 共享的行 stride，取自 `x_values.stride(0)` |
| `stride_col` | 共享的列 stride，取自 `x_values.stride(1)` |
| `output` | 预分配输出，shape、dtype 和布局与 `x_values` 相同 |
| `thread_block_row_size` | 每个 tile 处理的行数，constexpr |
| `thread_block_col_size` | 每个 tile 处理的列数，constexpr |
| `ops_func` | 运算类型，`"add"` 表示加法，`"mul"` 表示乘法；当前 ATK API 固定为 `"add"` |

## 输出

`output`：shape 为 `(M, N)`，dtype 与 `x_values` 相同。

当前 ATK API 的输出等价于：

```python
output = x_values + y_values
```

算子不修改两个输入，也不执行 broadcasting、offset 变换或 jagged/dense 格式转换。

## GPU 与 NPU 实现差异

GPU 端保留原始固定 tile 配置：

```text
thread_block_row_size = 32
thread_block_col_size = 32
grid = (ceil_div(M, 32) * ceil_div(N, 32),)
```

NPU 端保持 kernel 计算逻辑不变，并增加以 `(M, N)` 为 key 的 Triton autotune。候选配置为：

| 行 tile | 列 tile | `num_warps` | `num_stages` |
|--------:|--------:|------------:|-------------:|
| 16 | 16 | 2 | 2 |
| 32 | 32 | 4 | 2 |
| 32 | 32 | 4 | 3 |
| 64 | 64 | 8 | 3 |
| 32 | 64 | 4 | 2 |
| 64 | 32 | 4 | 2 |

NPU API 的 grid 会从 autotune 选中的 `thread_block_row_size` 和 `thread_block_col_size` 动态计算。

## 调用示例

以下示例直接调用 NPU 端 autotune kernel，与 NPU ATK API 的调用方式一致：

```python
import torch
import triton

from triton_jagged_tensor_ops import (
    jagged_jagged_elementwise_arithmetic_ops,
)

device = "npu"
x_values = torch.randn((128, 256), dtype=torch.float32, device=device)
y_values = torch.randn_like(x_values)
output = torch.empty_like(x_values)
M, N = x_values.shape


def grid(meta):
    return (
        triton.cdiv(M, meta["thread_block_row_size"])
        * triton.cdiv(N, meta["thread_block_col_size"]),
    )


jagged_jagged_elementwise_arithmetic_ops[grid](
    x_values,
    y_values,
    M,
    N,
    x_values.stride(0),
    x_values.stride(1),
    output,
    ops_func="add",
)

torch.testing.assert_close(output, x_values + y_values)
print(output.shape)  # torch.Size([128, 256])
```

如直接调用 GPU 端 kernel，请在 `gpu_atk_test/` 目录运行、将 `device` 改为 `"cuda"`，并显式传入两个 tile 参数及固定 grid：

```python
block_m, block_n = 32, 32
grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

jagged_jagged_elementwise_arithmetic_ops[grid](
    x_values,
    y_values,
    M,
    N,
    x_values.stride(0),
    x_values.stride(1),
    output,
    block_m,
    block_n,
    ops_func="add",
)
```

## 关键约束

- `x_values` 和 `y_values` 必须都是二维 tensor，且 `M > 0`、`N > 0`；当前 YAML 不生成空 tensor。
- 两个输入必须具有完全相同的 shape 和 dtype，不支持 broadcasting。
- kernel 只接收一组共享 stride，并使用该 stride 访问 `x_values`、`y_values` 和 `output`，因此三者必须具有相同的内存布局。当前 ATK 用例使用同 shape 的 contiguous tensor。
- 两个 values 矩阵应对应相同的 jagged offsets；offsets 不传入 kernel，算子无法在运行时检查这一条件。
- `thread_block_row_size` 和 `thread_block_col_size` 用于 `tl.arange`，应采用 Triton 支持的 2 的幂；当前固定值及 autotune 候选值均满足该条件。
- 当前 ATK API 仅测试加法。测试乘法时需要直接调用 kernel 并传入 `ops_func="mul"`，或扩展 API/YAML 暴露该参数。

## 目录结构

```text
experimental/triton/atk_test/triton_jagged_jagged_elementwise_arithmetic_ops/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_tensor_ops.py                              # GPU(CUDA) 端固定 tile kernel
│   └── triton_jagged_jagged_elementwise_arithmetic_ops_api.py   # GPU 端 ATK API
└── npu_atk_test/
    ├── triton_jagged_tensor_ops.py                              # NPU 端 autotune kernel
    ├── triton_jagged_jagged_elementwise_arithmetic_ops.yaml     # ATK 用例设计
    ├── generate_triton_jagged_jagged_elementwise_arithmetic_ops.py # shape/dtype 约束生成器
    ├── triton_jagged_jagged_elementwise_arithmetic_ops_api.py   # NPU 端 ATK API
    └── all_triton_jagged_jagged_elementwise_arithmetic_ops.json # 已生成的测试用例
```

## ATK 测试

测试在 GPU 与 NPU 两端协同进行：`gpu_atk_test/` 部署到 GPU 端，`npu_atk_test/` 部署到 NPU 端。以下生成及测试命令均在 NPU 端的 `npu_atk_test/` 目录中执行：

```bash
cd experimental/triton/atk_test/triton_jagged_jagged_elementwise_arithmetic_ops/npu_atk_test
```

### 远端 GPU 服务器启动监听

在 GPU 端进入 `gpu_atk_test/` 目录后执行：

```bash
atk server --devices 0 \
    --plugin_path triton_jagged_jagged_elementwise_arithmetic_ops_api.py
```

如 GPU server 不在默认地址或端口，请在 NPU 端执行测试时按 ATK 环境配置远端 host/port。

### 生成测试数据

```bash
atk case \
    -f triton_jagged_jagged_elementwise_arithmetic_ops.yaml \
    -p generate_triton_jagged_jagged_elementwise_arithmetic_ops.py
```

生成的用例通常位于：

```text
result/triton_jagged_jagged_elementwise_arithmetic_ops/json/
all_triton_jagged_jagged_elementwise_arithmetic_ops.json
```

本目录同时提供可直接复用的 `all_triton_jagged_jagged_elementwise_arithmetic_ops.json`。

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_triton_jagged_jagged_elementwise_arithmetic_ops.json \
    --task accuracy \
    -p triton_jagged_jagged_elementwise_arithmetic_ops_api.py
```

如需保存精度对比数据，在命令末尾增加 `--save_data profile`。

### 性能测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_triton_jagged_jagged_elementwise_arithmetic_ops.json \
    --task performance_device \
    -p triton_jagged_jagged_elementwise_arithmetic_ops_api.py
```

如需保存性能对比数据，在命令末尾增加 `--save_data profile`。
