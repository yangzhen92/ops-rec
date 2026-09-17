# _jagged_dense_bmm_broadcast_add

## 来源

算子源自 NVIDIA recsys-examples 的 Triton jagged kernel：

<https://github.com/NVIDIA/recsys-examples/blob/a247bd4baf4dc74c3e19b4f2ed1fd1b62eb75586/examples/hstu/ops/triton_ops/triton_jagged.py>

## 算子功能

该算子对 batch 内不同长度的序列执行矩阵乘和偏置广播：

```text
Out[seq_start:seq_end, :] = Jagged[seq_start:seq_end, :] @ Dense[b, :, :] + Bias[b, :]
```

`Jagged` 将各序列沿第一维拼接，`seq_offsets` 给出每个 batch 的起止位置。kernel 使用三维 grid 划分输出列、序列行和 batch，并以 fp32 累加。

## 输入与输出

| 参数 | 说明 |
|------|------|
| `seq_offsets` | int64 张量 `[B + 1]`，首元素为 0，末元素为 `L`，严格递增 |
| `Jagged` | 拼接后的输入 `[L, K]`，支持 fp32/fp16/bf16 |
| `Dense` | batch 矩阵 `[B, K, N]`，dtype 与 `Jagged` 相同 |
| `Bias` | 广播偏置 `[B, N]`，dtype 与 `Jagged` 相同 |
| `Out` | 输出 `[L, N]`，dtype 与 `Jagged` 相同 |
| `AUTOTUNE_MAX_SEQ_LEN` | batch 内最大序列长度 |
| `BLOCK_M/N/K` | autotune 选择的矩阵分块大小 |

## 使用示例

```python
import torch
import triton
from triton_jagged_dense_bmm_broadcast_add_kernel import jagged_dense_bmm_broadcast_add_kernel

offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
jagged = torch.randn((5, 32), dtype=torch.float16, device="npu")
dense = torch.randn((2, 32, 64), dtype=torch.float16, device="npu")
bias = torch.randn((2, 64), dtype=torch.float16, device="npu")
out = torch.empty((5, 64), dtype=torch.float16, device="npu")
grid = lambda meta: (triton.cdiv(64, meta["BLOCK_N"]), triton.cdiv(3, meta["BLOCK_M"]), 2)

jagged_dense_bmm_broadcast_add_kernel[grid](
    offsets, jagged, dense, bias, out, 3, 64, 32,
    jagged.stride(0), dense.stride(0), dense.stride(1), dense.stride(2),
    bias.stride(0), out.stride(0), HAS_BIAS=True, ALLOW_TF32=False,
)
```

## 关键约束

- 满足 `L >= B >= 1`，且每个 batch 至少包含一个 token。
- `Dense.shape[1] == K`，`Bias.shape == [B, N]`。
- 总元素数 `L*K + B*K*N + B*N + L*N` 不超过 `2^24`。
- NPU 生成器限制 grid 的 `coreDim <= 65535`。
- ATK API 根据输入 shape 确定性重建合法 offsets，保证 GPU/NPU 输入一致。

## 目录文件

```text
triton_jagged_dense_bmm_broadcast_add/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_dense_bmm_broadcast_add_kernel.py
│   └── triton_jagged_dense_bmm_broadcast_add_api.py
└── npu_atk_test/
    ├── triton_jagged_dense_bmm_broadcast_add_kernel.py
    ├── triton_jagged_dense_bmm_broadcast_add_api.py
    ├── triton_jagged_dense_bmm_broadcast_add.yaml
    ├── generate_triton_jagged_dense_bmm_broadcast_add.py
    └── triton_jagged_dense_bmm_broadcast_add_performance.json
```

## ATK 测试

### 生成精度测试数据

该命令读取 YAML 参数空间，并通过 generator 对齐 jagged/dense/bias shape，生成精度测试用例 JSON。

```bash
atk case -f triton_jagged_dense_bmm_broadcast_add.yaml -p generate_triton_jagged_dense_bmm_broadcast_add.py
```

### 启动远端 GPU server

该命令在 GPU 基准环境启动 ATK server，并从指定目录加载 GPU API 封装。

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

该命令由 NPU 侧连接 GPU server，分别执行两端 kernel，并比较广播矩阵乘输出的精度。

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c triton_jagged_dense_bmm_broadcast_add_performance.json --task accuracy \
    -p triton_jagged_dense_bmm_broadcast_add_api.py
```

### 性能测试

该命令读取性能用例 JSON，在 NPU 侧执行设备性能测试并记录 kernel 耗时。

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c triton_jagged_dense_bmm_broadcast_add_performance.json \
    --task performance_device -p triton_jagged_dense_bmm_broadcast_add_api.py
```
