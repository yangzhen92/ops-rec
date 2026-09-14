# `triton_jagged_jagged_elementwise_arithmetic_ops` 算子文档

## 来源

本目录的核心 kernel `jagged_jagged_elementwise_arithmetic_ops` 源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py#L16-L69>

## 功能说明

`jagged_jagged_elementwise_arithmetic_ops` 用于对两个 Jagged Tensor 已经展平的 values 矩阵执行逐元素加法或乘法。

设两个输入 values 的 shape 均为 `(M, N)`，则：

```text
output[m, n] = x_values[m, n] + y_values[m, n]  ops_func = "add"
output[m, n] = x_values[m, n] * y_values[m, n]  ops_func = "mul"
```

该算子只处理 values，不接收 Jagged offsets。调用方需要保证 `x_values` 与 `y_values` 来自逻辑结构一致的 Jagged Tensor，即两者具有相同的 offsets，且相同位置的 values 能够一一对应。

算子不会执行广播、Jagged/Dense 格式转换、offset 重排或跨 token 归约。当前 ATK API 固定使用 `ops_func="add"`，因此现有 ATK 用例验证的是逐元素加法；底层 kernel 同时支持乘法。

## 输入

| 参数 | 说明 |
|------|------|
| `x_values` | 第一个 Jagged Tensor 的展平 values，shape 为 `(M, N)` |
| `y_values` | 第二个 Jagged Tensor 的展平 values，shape 为 `(M, N)` |
| `ops_func` | 运算类型；`"add"` 表示逐元素加法，`"mul"` 表示逐元素乘法 |

`x_values` 和 `y_values` 支持 `fp32`、`fp16` 和 `bf16`。

## 输出

`output`：shape 为 `(M, N)`，dtype 与输入相同。

输出保持 values 的展平顺序，因此在两个输入共享相同 Jagged offsets 时，输出可以直接与同一组 offsets 组合成新的 Jagged Tensor。算子不修改 `x_values` 和 `y_values`。

## 计算示例

下面使用 NPU 上的 PyTorch 运算展示加法和乘法的等价语义。`offsets` 仅用于说明两个 values 属于相同的 Jagged 结构，不作为该算子的输入：

```python
import torch

device = "npu"

# 两个序列的长度分别为 2 和 3，values 的特征维度为 4
offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=device)
x_values = torch.randn((5, 4), dtype=torch.float32, device=device)
y_values = torch.randn_like(x_values)

add_output = x_values + y_values
mul_output = x_values * y_values

print(add_output.shape)  # torch.Size([5, 4])
print(mul_output.shape)  # torch.Size([5, 4])

# 输出继续复用同一组 Jagged offsets
for b in range(offsets.numel() - 1):
    begin = int(offsets[b].item())
    end = int(offsets[b + 1].item())
    print(add_output[begin:end].shape)
```

## 关键约束

- `x_values` 和 `y_values` 必须是二维 tensor，且行数 `M`、列数 `N` 均大于零。
- 两个输入的 shape、dtype、device 和内存布局必须一致，不支持 broadcasting。
- 输入与输出需要使用相同的二维 stride 布局；通常应传入 contiguous tensor。
- 两个输入必须对应相同的 Jagged offsets。offsets 不传入算子，因此无法在运行时检查该条件。
- `ops_func` 应使用 `"add"` 或 `"mul"`；当前 ATK API 固定执行 `"add"`。
- 加法输出等价于 `x_values + y_values`，乘法输出等价于 `x_values * y_values`。

## ATK 测试

以下命令均在 NPU 端的 `npu_atk_test` 目录下执行：

```bash
cd experimental/triton/atk_test/triton_jagged_jagged_elementwise_arithmetic_ops/npu_atk_test
```

### 生成测试数据

```bash
atk case -f triton_jagged_jagged_elementwise_arithmetic_ops.yaml \
    -p generate_triton_jagged_jagged_elementwise_arithmetic_ops.py
```

生成的用例配置文件为 `all_triton_jagged_jagged_elementwise_arithmetic_ops.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_triton_jagged_jagged_elementwise_arithmetic_ops.json \
    --task accuracy \
    -p triton_jagged_jagged_elementwise_arithmetic_ops_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_triton_jagged_jagged_elementwise_arithmetic_ops.json \
    --task performance_device \
    -p triton_jagged_jagged_elementwise_arithmetic_ops_api.py
```
