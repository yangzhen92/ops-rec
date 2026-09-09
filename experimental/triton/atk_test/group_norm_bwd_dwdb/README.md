# group_norm_bwd_dwdb

## 来源

`group_norm_bwd_dwdb` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

`group_norm_bwd_dwdb` 将 group norm 反向阶段产生的 partial weight/bias gradient 按行累加，得到最终的权重梯度 `FINAL_DW` 与偏置梯度 `FINAL_DB`。输入按 `[GROUP_N, num_heads]` 布局，每个 program 处理一列，结果为：

```text
FINAL_DW[col] = sum(DW[row, col] for row in range(GROUP_N))
FINAL_DB[col] = sum(DB[row, col] for row in range(GROUP_N))
```

### 输入

- `DW` `(GROUP_N, num_heads)` — partial dweight，`GROUP_N` 为 partial gradient 行数
- `DB` `(GROUP_N, num_heads)` — partial dbias，与 `DW` 同 shape

### 输出

- `FINAL_DW` `(num_heads,)` — 沿 `DW` 第 0 维规约得到的最终 dweight
- `FINAL_DB` `(num_heads,)` — 沿 `DB` 第 0 维规约得到的最终 dbias

### 约束

- `DW` / `DB` 必须是二维 contiguous tensor，`GROUP_N > 0`
- 内部先以 `float32` 累加再转换为输出 dtype
- 支持 `fp32`、`fp16`、`bf16`
- NPU 端通过 `@triton.autotune` 自动选择 `BLOCK_N`；GPU 端固定计算 `BLOCK_N`

## 目录结构

```text
group_norm_bwd_dwdb/
├── README.md
├── gpu_atk_test/
│   ├── group_norm_bwd_dwdb.py       # GPU 端 Triton kernel 实现
│   └── group_norm_bwd_dwdb_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── group_norm_bwd_dwdb.py       # NPU 端 Triton kernel 实现（autotune）
    ├── group_norm_bwd_dwdb.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_group_norm_bwd_dwdb.py  # ATK 用例生成器
    ├── group_norm_bwd_dwdb_api.py   # NPU 端 ATK API 执行文件
    └── group_norm_bwd_dwdb_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch
import triton

from group_norm_bwd_dwdb import _group_norm_bwd_dwdb

GROUP_N, num_heads = 1024, 8
device = "npu"
dw = torch.randn((GROUP_N, num_heads), dtype=torch.bfloat16, device=device)
db = torch.randn((GROUP_N, num_heads), dtype=torch.bfloat16, device=device)
final_dw = torch.empty((num_heads,), dtype=dw.dtype, device=device)
final_db = torch.empty((num_heads,), dtype=db.dtype, device=device)

grid = (num_heads,)
_group_norm_bwd_dwdb[grid](
    dw, db, final_dw, final_db,
    GROUP_N,
    BLOCK_N=128,
)
print(final_dw.shape)  # torch.Size([8])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/group_norm_bwd_dwdb_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f group_norm_bwd_dwdb.yaml -p generate_group_norm_bwd_dwdb.py
```

生成器会把 `DW` / `DB` 的 shape 统一为 `[GROUP_N, num_heads]`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_group_norm_bwd_dwdb.json --task accuracy \
    -p group_norm_bwd_dwdb_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c group_norm_bwd_dwdb_performance.json --task performance_device \
    -p group_norm_bwd_dwdb_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
