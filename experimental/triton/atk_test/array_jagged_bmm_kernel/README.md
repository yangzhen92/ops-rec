# array_jagged_bmm_kernel

## 来源

`array_jagged_bmm_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_bmm_jagged_out.py>

## 功能说明

`array_jagged_bmm_kernel` 实现 Array-Jagged 批量矩阵乘（jagged in, jagged out）：对每个 batch 做矩阵乘 `C[batch] = A[batch] @ B[batch]`，A 为该 batch 的方阵展平，B/C 为变长行数矩阵，结果按 batch 拼接输出。

### 输入

- `array_A` `(Σseq²,)` — 每 batch 为 `seq×seq` 方阵展平
- `jagged_B` `(Σseq, D)` — 每 batch 为 `seq×D`
- `offsets_am` / `offsets_bk` / `offsets_cm` `(B+1,)` — 三组 int32 offset（am 为平方量 cumsum，bk/cm 为线性量 cumsum）
- `max_seq_len` — 截断上限，取 `max(seq)` 时 kernel 不截断

### 输出

- `jagged_C` `(Σseq, D)` — 与 `jagged_B` 同 shape

### 约束

- `seq` 逐 batch 可变，`total_seq ≥ B`，`offsets` 单调递增且 `offsets[0] = 0`
- `array_A` 与 `jagged_B` dtype 一致，支持 `fp32`、`fp16`、`bf16`；offset 为 `int32`
- GPU 端固定 `BLOCK = 32×32×32` 且 `allow_tf32=False`；NPU 端通过 `@triton.autotune` 按 `[D, max_seq_len]` 自动选择 BLOCK

## 目录结构

```text
array_jagged_bmm_kernel/
├── README.md
├── gpu_atk_test/
│   ├── array_jagged_bmm_kernel.py       # GPU 端 Triton kernel 实现
│   └── array_jagged_bmm_kernel_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── array_jagged_bmm_kernel.py       # NPU 端 Triton kernel 实现（autotune）
    ├── array_jagged_bmm_kernel.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_array_jagged_bmm_kernel.py  # ATK 用例生成器
    ├── array_jagged_bmm_kernel_api.py   # NPU 端 ATK API 执行文件
    └── array_jagged_bmm_kernel_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch
import triton

from array_jagged_bmm_kernel import array_jagged_bmm_kernel

B, seq_len, D = 2, 2, 4
device = "npu"
array_A = torch.randn(B * seq_len * seq_len, dtype=torch.float32, device=device)
jagged_B = torch.randn(B * seq_len, D, dtype=torch.float32, device=device)
jagged_C = torch.zeros(B * seq_len, D, dtype=jagged_B.dtype, device=device)
offsets_am = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
offsets_bk = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
offsets_cm = offsets_bk.clone()

BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
grid = (triton.cdiv(D, BLOCK_SIZE_N), triton.cdiv(seq_len, BLOCK_SIZE_M), B)
array_jagged_bmm_kernel[grid](
    array_A,
    jagged_B,
    jagged_C,
    offsets_am,
    offsets_bk,
    offsets_cm,
    D,
    jagged_B.stride(0),
    jagged_B.stride(1),
    jagged_C.stride(0),
    jagged_C.stride(1),
    0,  # transpose
    seq_len,  # max_seq_len
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    False,  # allow_tf32
)
print(jagged_C.shape)  # torch.Size([4, 4])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/array_jagged_bmm_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f array_jagged_bmm_kernel.yaml -p generate_array_jagged_bmm_kernel.py
```

生成器按 `(total_seq, B)` 做确定性变长切分，覆盖 `array_A = [Σseq²]`、三个 `offsets = [B+1]`；API 侧用同一套切分逻辑重建 offsets，保证两端一致。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_array_jagged_bmm_kernel.json --task accuracy \
    -p array_jagged_bmm_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c array_jagged_bmm_kernel_performance.json --task performance_device \
    -p array_jagged_bmm_kernel_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
