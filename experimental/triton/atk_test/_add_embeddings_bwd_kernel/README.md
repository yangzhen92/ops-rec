# `_add_embeddings_bwd_kernel` 算子文档

## 来源

`_add_embeddings_bwd_kernel` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 算子功能

`_add_embeddings_bwd_kernel` 用于将 position/timestamp embeddings 的**反向梯度**按归并 key 做 scatter-add（原子累加），每组内同一 key 的 token 梯度求和。

输入为展平 token 的上游梯度 `In` 与排序分段的 `KeyInds`/`ValueInds` 索引对。对相同 key 的分组，组内 `In[ValueInds[i]]` 各行先本地累加，再 `atomic_add`（relaxed）汇入 `Out[KeyInds[i]]`。

`KeyInds`/`ValueInds` 由 `num_keys` 与 `jagged_size` 确定性推导（`ValueInds[i] = i`，`KeyInds[i] = i * num_keys // jagged_size`），保证 GPU/NPU 两端输入一致。

`Out` 以零初始化，`atomic_add` 累加要求每次调用从零开始。

## 输入

| 参数 | 说明 |
|------|------|
| `In` | 展平 token 的上游梯度，shape `(jagged_size, D)`，dtype 支持 fp32/fp16/bf16 |
| `KeyInds` | 每个 token 归并到的 key（已排序分段，值域 `[0, num_keys)`） |
| `ValueInds` | 被累加的行索引，`ValueInds[i] = i` |
| `num_keys` | 输出 key 数（`Out` 行数），要求 `jagged_size >= num_keys` |
| `Out` | 输出梯度（embedding 表各行梯度），shape `(num_keys, D)`，以零初始化 |
| `BLOCK_D` | 列块大小（constexpr），取 `next_power_of_2(D)` |
| `BLOCK` | 行块大小（constexpr） |

## 输出

`Out`：形状 `(num_keys, D)`，其中 `num_keys` 由输入指定，dtype 与 `In` 相同。`atomic_add` 累加要求每次调用从零开始（API 在 `__call__` 前 `zero_()`）。

## 使用样例

### 直接调用 kernel

```python
import torch
import triton
from _add_embeddings_bwd_kernel import _add_embeddings_bwd_kernel

in_data = torch.randn(200, 16, dtype=torch.float32, device="npu")
num_keys = 4
key_inds = ((torch.arange(200) * num_keys) // 200).to(torch.int32).to("npu")
value_inds = torch.arange(200, dtype=torch.int32, device="npu")
out = torch.zeros(4, 16, dtype=torch.float32, device="npu")

grid = (triton.cdiv(200, 128),)
_add_embeddings_bwd_kernel[grid](
    In=in_data,
    KeyInds=key_inds,
    ValueInds=value_inds,
    Out=out,
    AUTOTUNE_MAX_SEQ_LEN=1024,
    B=1,
    AUTOTUNE_B=1,
    D=16,
    jagged_size=200,
    stride_in=in_data.stride(0),
    stride_on=out.stride(0),
    BLOCK_D=16,
    BLOCK=128,
)
```

## 关键约束

- In.ndim == 2 且 jagged_size >= num_keys（每个 key 至少被一个 token 寻址）。
- Out 以 torch.zeros 初始化，atomic_add 需要零起点；多次调用前需归零。
- KeyInds 已排序（API 按 i * num_keys // jagged_size 确定性生成），组内累加才成立。

## 目录文件

```text
_add_embeddings_bwd_kernel/
├── README.md
├── gpu_atk_test/
│   ├── _add_embeddings_bwd_kernel.py              # 原版算子 kernel 实现
│   └── _add_embeddings_bwd_kernel_api.py          # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── _add_embeddings_bwd_kernel.py              # NPU 适配优化后算子 kernel 实现
    ├── _add_embeddings_bwd_kernel_api.py          # NPU 端 ATK 自定义 API 封装
    ├── _add_embeddings_bwd_kernel.yaml            # ATK 算子用例参数
    ├── generate_add_embeddings_bwd_kernel.py      # 限制参数生成脚本
    └── _add_embeddings_bwd_kernel_performence.json        # 性能测试样例数据
 ```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/_add_embeddings_bwd_kernel_performence.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成精度测试数据

```bash
atk case -f _add_embeddings_bwd_kernel.yaml -p generate_add_embeddings_bwd_kernel.py
```

生成的精度测试用例位于 `result/_add_embeddings_bwd_kernel/json/` 下，文件名为 `all__add_embeddings_bwd_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__add_embeddings_bwd_kernel.json --task accuracy \
    -p _add_embeddings_bwd_kernel_api.py
```

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c _add_embeddings_bwd_kernel_performence.json --task performance_device \
    -p _add_embeddings_bwd_kernel_api.py
```
