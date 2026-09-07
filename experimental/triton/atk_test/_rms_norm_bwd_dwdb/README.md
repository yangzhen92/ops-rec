# _rms_norm_bwd_dwdb

## 来源

`_rms_norm_bwd_dwdb` 是 RMSNorm 反向计算中的 dweight 规约算子，用 Triton 实现。它的分块归约方式参考了 Triton LayerNorm 教程中 `_layer_norm_bwd_dwdb` 的 partial gradient reduction：

<https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>

## 功能说明

`_rms_norm_bwd_dwdb` 将 RMSNorm 反向阶段产生的 partial weight gradient 沿第 0 维累加，得到最终的权重梯度 `FINAL_DW`。对每个特征列 `d`，计算结果为：

```text
FINAL_DW[d] = sum(DW[n, d] for n in range(N))
```

GPU 和 NPU 端都按特征维度分块启动 Triton program，每个 program 负责一段连续的列，并在循环中加载所有行后完成列归约。累加在 `float32` 中进行，写回时转换为输出 tensor 的 dtype。

### 输入

- `DW` `(N, D)` — RMSNorm 反向阶段产生的 partial dweight，`N` 为 partial gradient 行数，`D` 为特征维度

### 输出

- `FINAL_DW` `(D,)` — 沿 `DW` 第 0 维规约得到的最终 dweight

### 约束

- `DW` 必须是二维 contiguous tensor，`N > 0`、`D > 0`
- `FINAL_DW.shape[0] = DW.shape[1] = D`
- `DW` 与 `FINAL_DW` 使用相同 dtype
- 支持 `fp32`、`fp16`、`bf16`
- 输出等价于 `DW.sum(dim=0)`，但 kernel 内部先以 `float32` 累加再转换为输出 dtype

## 目录结构

```text
experimental/triton/atk_test/_rms_norm_bwd_dwdb/
├── README.md
├── gpu_atk_test/
│   ├── _rms_norm_bwd_dwdb_kernel.py   # GPU(CUDA) 端 Triton kernel 实现
│   └── _rms_norm_bwd_dwdb_api.py      # GPU(CUDA) 端 ATK API 执行文件
└── npu_atk_test/
    ├── _rms_norm_bwd_dwdb_kernel.py   # NPU(Triton) kernel 实现
    ├── _rms_norm_bwd_dwdb.yaml        # ATK 用例设计（dtype/shape/约束）
    ├── generate_rms_norm_bwd_dwdb.py  # ATK 用例生成器
    ├── _rms_norm_bwd_dwdb_api.py      # NPU(Triton) 端 ATK API 执行文件
    └── all__rms_norm_bwd_dwdb.json    # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入 GPU kernel 调用（与 GPU ATK API 中的用法一致）：

```python
import torch
import triton

from _rms_norm_bwd_dwdb_kernel import _rms_norm_bwd_dwdb_kernel

N, D = 1024, 128
device = "cuda"
dw = torch.randn((N, D), dtype=torch.bfloat16, device=device)
final_dw = torch.empty((D,), dtype=dw.dtype, device=device)

BLOCK_N, BLOCK_D = 32, 32
grid = (triton.cdiv(D, BLOCK_D),)
_rms_norm_bwd_dwdb_kernel[grid](
    dw,
    final_dw,
    N,
    D,
    BLOCK_N=BLOCK_N,
    BLOCK_D=BLOCK_D,
)
print(final_dw.shape)  # torch.Size([128])
```

NPU 端 API 使用同样的输入输出语义，并由 Triton autotune 根据 `N`、`D` 选择 `BLOCK_N` 和 `BLOCK_D`。

## ATK 测试

ATK 脚本目录：`experimental/triton/atk_test/_rms_norm_bwd_dwdb/npu_atk_test/`（相对仓库根目录），以下命令均在该目录下执行：

```bash
cd experimental/triton/atk_test/_rms_norm_bwd_dwdb/npu_atk_test
```

### 远端 NPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path _rms_norm_bwd_dwdb_api.py
```

### 3.1 生成测试数据

```bash
atk case -f _rms_norm_bwd_dwdb.yaml -p generate_rms_norm_bwd_dwdb.py
```

生成器会统一 `DW` 的第二维与 `FINAL_DW` 的第一维，并保证两者 dtype 一致。

### 3.2 精度对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all__rms_norm_bwd_dwdb.json --task accuracy -p _rms_norm_bwd_dwdb_api.py
```

注：如需保存精度对比数据，需要在命令后加 `--save_data profile`。

### 3.3 性能对比

```bash
atk node --backend npu --devices 0 node --backend npu --devices 0 task -c all__rms_norm_bwd_dwdb.json --task performance -p _rms_norm_bwd_dwdb_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
