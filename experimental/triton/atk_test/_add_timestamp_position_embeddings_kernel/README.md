# `_add_timestamp_position_embeddings_kernel` 算子文档

## 来源

`_add_timestamp_position_embeddings_kernel` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 算子功能

`_add_timestamp_position_embeddings_kernel` 用于把 timestamp/position embeddings 叠加到序列 embedding 上（HSTU 的时间/位置编码融合）：

- `Out[n] = SeqEmb[n] + PosEmb[pos_inds[n]] + TsEmb[ts_inds[n]]`
- 位置索引 `pos_inds` 按“倒数位置”计算（离 query 越近的行索引越小），并受 `max_contextual_seq_len` 与 `max_pos_ind` 钳位。
- 时间索引 `ts_inds` 由 `query_time - ts + time_delta` 经 `sqrt`（或 `log`）分桶得到，桶数上限为 `num_time_buckets`。
- `TRAINING=True` 时把计算出的 `pos_inds` / `ts_inds` 写回 `PosInds` / `TsInds`，供反向复用。

## 输入

| 参数 | 说明 |
|------|------|
| `SeqEmb` | 展平 token 的序列 embedding，shape `(jagged_size, D)`，dtype 支持 fp32/fp16/bf16 |
| `Offsets` | 每个 batch 的起始行偏移，shape `(B+1,)` |
| `Lengths` | 每个 batch 的序列长度，shape `(B,)` |
| `PosEmb` | 位置编码查找表，shape `(N_p, D)`，`N_p = max_contextual_seq_len + max_seq_len` |
| `TsEmb` | 时间编码查找表，shape `(N_t, D)`，`N_t = num_time_buckets + 1` |
| `TS` | 每个 token 的时间戳，shape `(jagged_size,)` |
| `PosInds` / `TsInds` | 位置/时间桶索引输出（TRAINING=True 时写回），shape `(jagged_size,)` |
| `NumTargets` | 每个 batch 的 target 数（HAS_MULTIPLE_TARGETS=True 时使用） |
| `num_time_buckets` | 时间桶数量（固定 2048） |
| `max_contextual_seq_len` | 位置桶基准长度 |
| `BLOCK_D` / `BLOCK_N` | 列/行块大小（constexpr） |

## 输出

`Out`：形状 `(jagged_size, D)`，dtype 与 `SeqEmb` 相同。kernel 逐 token 全量覆盖写，无需归零。

## 使用样例

### 直接调用 kernel

```python
import torch
import triton
from _add_timestamp_position_embeddings_kernel import _add_timestamp_position_embeddings_kernel

seq_emb = torch.randn(200, 16, dtype=torch.float32, device="npu")
offsets = torch.tensor([0, 100, 200], dtype=torch.int32, device="npu")
lengths = torch.tensor([100, 100], dtype=torch.int32, device="npu")
pos_emb = torch.randn(164, 16, dtype=torch.float32, device="npu")
ts_emb = torch.randn(2049, 16, dtype=torch.float32, device="npu")
ts = torch.rand(200, dtype=torch.float32, device="npu")
pos_inds = torch.empty(200, dtype=torch.int32, device="npu")
ts_inds = torch.empty(200, dtype=torch.int32, device="npu")
num_targets = torch.zeros(2, dtype=torch.int32, device="npu")
out = torch.empty(200, 16, dtype=torch.float32, device="npu")

grid = (2, triton.cdiv(100, 64))
_add_timestamp_position_embeddings_kernel[grid](
    SeqEmb=seq_emb,
    Offsets=offsets,
    Lengths=lengths,
    PosEmb=pos_emb,
    TsEmb=ts_emb,
    Out=out,
    TS=ts,
    PosInds=pos_inds,
    TsInds=ts_inds,
    NumTargets=num_targets,
    AUTOTUNE_MAX_SEQ_LEN=64,
    D=16,
    num_time_buckets=2048,
    time_bucket_increments=1.0,
    time_bucket_scale=1.0,
    time_delta=0.0,
    max_contextual_seq_len=64,
    max_pos_ind=164,
    stride_sn=seq_emb.stride(0),
    stride_pn=pos_emb.stride(0),
    stride_tn=ts_emb.stride(0),
    stride_on=out.stride(0),
    TRAINING=True,
    HAS_MULTIPLE_TARGETS=False,
    INTERLEAVE_TARGETS=False,
    TIME_BUCKET_FN="sqrt",
    BLOCK_D=16,
    BLOCK_N=64,
)
```

## 关键约束

- `SeqEmb.ndim == 2` 且 `jagged_size >= num_batches`（每个 batch 至少一个 token）。
- 每序列长度 `<= 4096`：保证 `(seq_len-1-n)^2` 在 fp32 中精确表示，`sqrt` 桶索引确定性成立。
- `TIME_BUCKET_FN` 固定 `"sqrt"`（确定性要求，API 默认值）。
- `TRAINING=True` 时 `PosInds` / `TsInds` 会被覆盖写。

## 目录文件

```text
_add_timestamp_position_embeddings_kernel/
├── README.md
├── gpu_atk_test/
│   ├── _add_timestamp_position_embeddings_kernel.py       # 原版算子 kernel 实现
│   └── _add_timestamp_position_embeddings_kernel_api.py   # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── _add_timestamp_position_embeddings_kernel.py       # NPU 适配（autotune）kernel 实现
    ├── _add_timestamp_position_embeddings_kernel_api.py   # NPU 端 ATK 自定义 API 封装
    ├── _add_timestamp_position_embeddings_kernel.yaml     # ATK 算子用例参数
    ├── generate_add_timestamp_position_embeddings_kernel.py  # 限制参数生成脚本
    └── all__add_timestamp_position_embeddings_kernel.json    # 测试样例数据
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 精度测试采用单基准模式（acc: single_bm），性能测试未配置（perf: not_key）。

### 生成精度测试数据

```bash
atk case -f _add_timestamp_position_embeddings_kernel.yaml -p generate_add_timestamp_position_embeddings_kernel.py
```

生成的精度测试用例位于 `result/_add_timestamp_position_embeddings_kernel/json/` 下，文件名为 `all__add_timestamp_position_embeddings_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all__add_timestamp_position_embeddings_kernel.json --task accuracy \
    -p _add_timestamp_position_embeddings_kernel_api.py
```
