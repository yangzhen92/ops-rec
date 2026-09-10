# triton_jagged_jagged_bmm_reduce_sum

## 来源

`triton_jagged_jagged_bmm_reduce_sum` 算子源自 NVIDIA recsys-examples 中的 Triton jagged kernel：

<https://github.com/NVIDIA/recsys-examples/blob/v25.05/examples/hstu/ops/triton_ops/triton_jagged.py>

## 算子功能

该算子对 batch 内两组共享 jagged 维度的矩阵执行分段矩阵乘。每个 batch 的序列范围由 `seq_offsets` 指定：

```text
start = seq_offsets[b]
end = seq_offsets[b + 1]
Out[b, m, n] = sum(
    JaggedA[k, m] * JaggedB[k, n],
    k in [start, end),
)
```

`JaggedA` 和 `JaggedB` 将各 batch 的可变长序列沿第一维拼接，两者共享总长度 `L = sum(K_b)`。kernel 使用三维 grid 划分 batch、输出行和输出列，并以 fp32 完成矩阵乘累加。

kernel 还支持在 `REDUCE_JAGGEDB=True` 时计算：

```text
ReduceOut[b, n] = sum(JaggedB[k, n], k in [start, end))
```

当前 GPU/NPU ATK API 均设置 `REDUCE_JAGGEDB=False`，因此测试只返回 `Out`。

## 输入

| 参数 | 说明 |
|------|------|
| `seq_offsets` | int64 一维张量，shape 为 `[B + 1]` |
| `JaggedA` | 左输入，shape 为 `[L, M]`，支持 fp32/fp16/bf16 |
| `JaggedB` | 右输入，shape 为 `[L, N]`，dtype 与 `JaggedA` 相同 |
| `Out` | 输出张量指针，shape 为 `[B, M, N]` |
| `ReduceOut` | 可选归约输出指针，shape 为 `[B, N]` |
| `M` | `JaggedA` 的特征维度 |
| `N` | `JaggedB` 的特征维度 |
| `AUTOTUNE_MAX_SEQ_LEN` | batch 内最大序列长度，参与 NPU autotune key |
| `stride_ak` | `JaggedA` 第一维 stride |
| `stride_bk` | `JaggedB` 第一维 stride |
| `stride_ob/om/on` | `Out` 的 batch、M、N 维 stride |
| `stride_orb/orn` | `ReduceOut` 的 batch、N 维 stride |
| `REDUCE_JAGGEDB` | constexpr，是否生成 `ReduceOut` |
| `ALLOW_TF32` | constexpr，是否允许 `tl.dot` 使用 TF32 |
| `BLOCK_M/N/K` | constexpr，矩阵乘的 M、N、K 分块大小 |

ATK YAML 中配置的输入顺序为 `seq_offsets`、`JaggedA`、`JaggedB`、`ReduceOut`。包装层使用输入的 shape 和 dtype，但会确定性重建合法 offsets，并重新分配 kernel 使用的 `ReduceOut`。

## 输出

`Out`：shape 为 `[B, M, N]`，dtype 与 `JaggedA` 相同。

当直接调用 kernel 且设置 `REDUCE_JAGGEDB=True` 时，还会写入 `ReduceOut[B, N]`；当前 ATK 用例未启用此输出。

## 使用示例

### 直接调用 NPU autotune kernel

```python
import torch
import triton

from triton_jagged_jagged_bmm_reduce_sum_kernel import _jagged_jagged_bmm_reduce_sum

offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
jagged_a = torch.randn((5, 32), dtype=torch.float16, device="npu")
jagged_b = torch.randn((5, 64), dtype=torch.float16, device="npu")
out = torch.empty((2, 32, 64), dtype=jagged_a.dtype, device="npu")
reduce_out = torch.empty((2, 64), dtype=jagged_a.dtype, device="npu")

grid = lambda meta: (
    2,
    triton.cdiv(32, meta["BLOCK_M"]),
    triton.cdiv(64, meta["BLOCK_N"]),
)

_jagged_jagged_bmm_reduce_sum[grid](
    offsets,
    jagged_a,
    jagged_b,
    out,
    reduce_out,
    32,
    64,
    3,
    jagged_a.stride(0),
    jagged_b.stride(0),
    out.stride(0),
    out.stride(1),
    out.stride(2),
    reduce_out.stride(0),
    reduce_out.stride(1),
    REDUCE_JAGGEDB=False,
    ALLOW_TF32=False,
)
```

## GPU 与 NPU 实现

- GPU API 使用固定配置 `BLOCK_M=64`、`BLOCK_N=64`、`BLOCK_K=32`。
- NPU kernel 使用 `@triton.autotune`，在 `BLOCK_M/BLOCK_N={64,128}`、`BLOCK_K={32,64}` 的 8 种组合中选择配置。
- NPU autotune key 为 `M`、`N` 和 `AUTOTUNE_MAX_SEQ_LEN`。
- 两端 API 使用相同算法确定性重建 offsets，确保每段至少包含一个 token。
- ATK CPU golden 可能将 fp32 输入提升为 fp64；API 会在启动 kernel 前转换回 fp32，因为 `tl.dot` 不支持 fp64。

## Generator shape 修正

YAML 会独立生成各输入 shape，generator 将其修正为满足 kernel 关系的 `B/L/M/N`：

1. 根据请求的 `L` 选择小规模或大规模的 M/N 候选集合。
2. 根据 offsets 长度和 M/N 确定 batch size，并保证 `B <= L`。
3. 根据总元素数上限压缩 `L`，避免测试数据占用过多内存。
4. 调整 M、N 或 B，使 NPU grid 的 `coreDim` 不超过限制。
5. 写回 `JaggedA[L,M]`、`JaggedB[L,N]`、`ReduceOut[B,N]` 和 `seq_offsets[B+1]`。
6. 强制三个浮点输入使用相同 dtype。

## 关键约束

- 满足 `L >= B >= 1`，每个 batch 至少包含一个 token。
- offsets 首元素为 0、末元素为 L，并且严格递增。
- `JaggedA.shape[0] == JaggedB.shape[0]`。
- `ReduceOut.shape == [B, N]`。
- `JaggedA`、`JaggedB` 和 `ReduceOut` 的 dtype 必须一致。
- 总元素数 `L*M + L*N + B*M*N + B*N` 不超过 `2^24`。
- generator 将 `L` 限制在 65536 以内。
- NPU 保守校验满足 `B * cdiv(M, 64) * cdiv(N, 64) <= 65535`。
- 空序列由 kernel 写入全零输出，但当前 API 构造的每个序列均非空。

## 目录文件

```text
triton_jagged_jagged_bmm_reduce_sum/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_jagged_bmm_reduce_sum_kernel.py
│   └── triton_jagged_jagged_bmm_reduce_sum_api.py
└── npu_atk_test/
    ├── triton_jagged_jagged_bmm_reduce_sum_kernel.py
    ├── triton_jagged_jagged_bmm_reduce_sum_api.py
    ├── triton_jagged_jagged_bmm_reduce_sum.yaml
    ├── generate_triton_jagged_jagged_bmm_reduce_sum.py
    └── triton_jagged_jagged_bmm_reduce_sum_performance.json
```

## ATK 测试

测试由 GPU server 与 NPU client 协同执行。精度用例通过 YAML 和 generator 生成，性能用例使用已提供的一行 JSON 文件。

### 生成精度测试数据

该命令读取 YAML 参数空间，并通过 generator 修正 jagged 输入之间的 shape 和 dtype 关系，生成精度测试用例 JSON。

```bash
atk case -f triton_jagged_jagged_bmm_reduce_sum.yaml \
    -p generate_triton_jagged_jagged_bmm_reduce_sum.py
```

生成结果位于 `result/triton_jagged_jagged_bmm_reduce_sum/json/`，用例文件名为 `all_triton_jagged_jagged_bmm_reduce_sum.json`。

### 启动远端 GPU server

该命令在 GPU 基准环境启动 ATK server，并从指定目录加载 GPU API 封装。

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

该命令由 NPU 侧连接 GPU server，分别执行两端 kernel，并比较分段矩阵乘输出的精度。

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_triton_jagged_jagged_bmm_reduce_sum.json --task accuracy \
    -p triton_jagged_jagged_bmm_reduce_sum_api.py
```

### 性能测试

该命令读取性能用例 JSON，在 NPU 侧执行设备性能测试并记录 kernel 耗时。

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c triton_jagged_jagged_bmm_reduce_sum_performance.json \
    --task performance_device -p triton_jagged_jagged_bmm_reduce_sum_api.py
```
