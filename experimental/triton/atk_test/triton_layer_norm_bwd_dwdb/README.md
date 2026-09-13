# triton_layer_norm_bwd_dwdb

## 来源

`_layer_norm_bwd_dwdb` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_layer_norm.py>

## 功能说明

LayerNorm 反向过程通常先按输入分块计算权重和偏置的局部梯度。该 kernel 对二维 partial
gradient 沿第一维规约，得到最终的一维 `dW` 与 `dB`：

```text
FINAL_DW[d] = sum(DW[n, d], n=0..N-1)
FINAL_DB[d] = sum(DB[n, d], n=0..N-1)
```

每个 Triton program 负责一段特征列，按 `BLOCK_N` 分块读取规约行，并使用 mask 处理
`N` 或 `D` 不能整除 block 的尾部数据。

### 输入与输出

- `DW` `(N, D)`：LayerNorm 反向前序 kernel 生成的局部权重梯度。
- `DB` `(N, D)`：LayerNorm 反向前序 kernel 生成的局部偏置梯度。
- `FINAL_DW` `(D,)`：最终权重梯度输出缓冲区。
- `FINAL_DB` `(D,)`：最终偏置梯度输出缓冲区。
- `N`：规约行数，API 以 `DW.shape[0]` 为准。

### 约束

- `DW` 与 `DB` 必须是 shape 相同的二维 tensor。
- `FINAL_DW` 与 `FINAL_DB` 必须是一维 tensor，长度等于 `DW.shape[1]`。
- 四个 tensor 使用相同 dtype，支持 `fp32`、`fp16`、`bf16`。
- generator 会统一各输入的 shape 与 dtype，并限制 `N * D` 以避免内存占用过大。

## 目录结构

```text
experimental/triton/atk_test/triton_layer_norm_bwd_dwdb/
├── README.md
├── gpu_atk_test/
│   ├── triton_layer_norm_bwd_dwdb_kernel.py
│   └── triton_layer_norm_bwd_dwdb_api.py
└── npu_atk_test/
    ├── triton_layer_norm_bwd_dwdb_kernel.py
    ├── triton_layer_norm_bwd_dwdb_api.py
    ├── triton_layer_norm_bwd_dwdb.yaml
    ├── generate_triton_layer_norm_bwd_dwdb.py
    └── triton_layer_norm_bwd_dwdb_performance.json
```

- `*_kernel.py`：实现 partial `DW/DB` 的并行规约；NPU 版本通过 autotune 选择 block。
- `*_api.py`：校验输入、搬移 tensor、计算 grid 并启动 kernel。
- `*.yaml`：定义 ATK 随机精度用例。
- `generate_*.py`：保证四个 tensor 的维度及 dtype 关系满足 kernel 约束。
- `*_performance.json`：保存固定性能用例。

## 调用示例

```python
from triton_layer_norm_bwd_dwdb_api import LayerNormBwdDwdbApi

api = LayerNormBwdDwdbApi()
api.init_by_input_data(input_data)
final_dw, final_db = api()
```

`input_data` 由 ATK 构造，实际测试时通常无需手工实例化 API。

## ATK 测试

### 1. GPU 端启动服务

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

该命令在 GPU 设备 `0` 启动标杆服务并加载 GPU API 与 kernel。

### 2. NPU 端生成精度用例

```bash
atk case -f triton_layer_norm_bwd_dwdb.yaml -p generate_triton_layer_norm_bwd_dwdb.py
```

该命令按 YAML 生成随机输入，再由 generator 对齐 `DW/DB/FINAL_DW/FINAL_DB`，输出
`all_triton_layer_norm_bwd_dwdb.json`。

### 3. NPU 端执行精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_layer_norm_bwd_dwdb.json --task accuracy -p triton_layer_norm_bwd_dwdb_api.py
```

该命令分别执行 NPU 与远端 GPU 实现，并比较 `FINAL_DW`、`FINAL_DB` 的数值误差。

### 4. NPU 端执行性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_layer_norm_bwd_dwdb_performance.json --task performance_device -p triton_layer_norm_bwd_dwdb_api.py
```

该命令使用固定性能用例测量两端设备耗时。需要保留中间数据时，可在任务命令末尾增加
`--save_data profile`。
