# triton_ln_mul_dropout_bwd_dwdb

## 来源

`_ln_mul_dropout_bwd_dwdb` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/a247bd4baf4dc74c3e19b4f2ed1fd1b62eb75586/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

该 kernel 是 LayerNorm、逐元素乘法与 Dropout 融合反向过程的第二阶段。前序
`_ln_mul_dropout_bwd_dx_du` kernel 按 tile 生成 partial 权重和偏置梯度，本 kernel
沿 tile 维执行规约：

```text
FINAL_DW[d] = sum(DW[n, d], n=0..N-1)
FINAL_DB[d] = sum(DB[n, d], n=0..N-1)
```

每个 Triton program 处理一段特征列，循环加载多行 partial gradient 并累加。NPU 版本
通过 autotune 在不同 `BLOCK_N` 和 `BLOCK_D` 组合中选择执行配置。

### 输入与输出

- `DW` `(N, D)`：前序融合反向 kernel 生成的局部权重梯度。
- `DB` `(N, D)`：前序融合反向 kernel 生成的局部偏置梯度。
- `FINAL_DW` `(D,)`：最终 LayerNorm 权重梯度。
- `FINAL_DB` `(D,)`：最终 LayerNorm 偏置梯度。
- `N`：需要规约的 partial 行数。

### 约束

- `DW` 与 `DB` 必须是 shape 相同的二维 tensor。
- 两个最终输出缓冲区的长度必须等于特征维 `D`。
- 输入输出 tensor 必须使用相同 dtype，支持 `fp32`、`fp16`、`bf16`。
- generator 会统一 shape 和 dtype，并控制 `N * D` 的最大元素数。

## 目录结构

```text
experimental/triton/atk_test/ln_mul_dropout_bwd_dwdb/
├── README.md
├── gpu_atk_test/
│   ├── triton_ln_mul_dropout_bwd_dwdb_kernel.py
│   └── triton_ln_mul_dropout_bwd_dwdb_api.py
└── npu_atk_test/
    ├── triton_ln_mul_dropout_bwd_dwdb_kernel.py
    ├── triton_ln_mul_dropout_bwd_dwdb_api.py
    ├── triton_ln_mul_dropout_bwd_dwdb.yaml
    ├── generate_triton_ln_mul_dropout_bwd_dwdb.py
    └── triton_ln_mul_dropout_bwd_dwdb_performance.json
```

- `*_kernel.py`：实现 partial 权重/偏置梯度规约。
- `*_api.py`：校验 ATK 输入，准备 grid 并启动 Triton kernel。
- `*.yaml`：定义随机精度用例的 dtype、shape 和取值范围。
- `generate_*.py`：修正独立生成输入之间的 shape 与 dtype 依赖。
- `*_performance.json`：提供固定性能测试用例。

## 调用示例

```python
import torch
import triton

from triton_ln_mul_dropout_bwd_dwdb_kernel import _ln_mul_dropout_bwd_dwdb

n, d = 128, 1024
dw = torch.randn((n, d), device="cuda", dtype=torch.float16)
db = torch.randn_like(dw)
final_dw = torch.empty((d,), device="cuda", dtype=dw.dtype)
final_db = torch.empty_like(final_dw)
block_n, block_d = 64, 32
grid = (triton.cdiv(d, block_d),)

_ln_mul_dropout_bwd_dwdb[grid](
    dw, db, final_dw, final_db, n, d, BLOCK_N=block_n, BLOCK_D=block_d
)
```

## ATK 测试

### 1. GPU 端启动服务

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

该命令在 GPU 设备 `0` 启动远端标杆服务，并加载 GPU 目录中的 API 与 kernel。

### 2. NPU 端生成精度用例

```bash
atk case -f triton_ln_mul_dropout_bwd_dwdb.yaml -p generate_triton_ln_mul_dropout_bwd_dwdb.py
```

该命令根据 YAML 采样输入，generator 随后统一 partial tensor 与输出缓冲区的维度，生成
`all_triton_ln_mul_dropout_bwd_dwdb.json`。

### 3. NPU 端执行精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_ln_mul_dropout_bwd_dwdb_performance.json --task accuracy -p triton_ln_mul_dropout_bwd_dwdb_api.py
```

该命令在 NPU 本地和远端 GPU 分别执行规约，并比较最终 `DW/DB` 输出。

### 4. NPU 端执行性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_ln_mul_dropout_bwd_dwdb_performance.json --task performance_device -p triton_ln_mul_dropout_bwd_dwdb_api.py
```

该命令使用固定性能用例统计两端设备耗时。需要保存中间结果时，在任务命令末尾增加
`--save_data profile`。
