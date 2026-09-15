# jagged_flash_attention_basic_kernel

## 来源

`jagged_flash_attention_basic_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/5d0a3d9868e04a0e0dd30906c79a96a001b85f68/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_flash_attention_basic.py#L18>

## 功能说明

`jagged_flash_attention_basic_kernel` 实现 Jagged Flash Attention 前向计算：对每个 batch 的变长序列计算 `o, lse = flash_attention(q, k, v, offset, max_seq_len)`，采用 FlashAttention 分块在线 softmax，输出注意力结果 `o` 与 log-sum-exp `lse`。

### 输入

- `q` `(total_seq, D)` / `v` `(total_seq, D)` / `k` `(D, total_seq)`（转置形式）
- `offset` `(B+1,)` int64 jagged offset；`max_seq_len` 最大序列长度
- `use_mask` 是否因果 mask；`allow_tf32` 是否允许 TF32；`BLOCK_SIZE_M/N/D` 分块大小

### 输出

- `o` `(total_seq, D)` — 注意力输出；`lse` `(total_seq,)` — log-sum-exp

### 约束

- q/k/v dtype 一致，支持 `fp32`、`fp16`、`bf16`；`offset` 为 `int64`；`D` 取 2 的幂次方；`total_seq ≥ B`（B=4~10）；`offset` 首 0 尾 `total_seq`
- GPU 端固定 `BLOCK = 32×32×32`；NPU 端通过 `@triton.autotune` 按 `[max_seq_len, D]` 自动选择 BLOCK（16/32 档）；NPU 端输入统一转 fp32 编译，输出转回原始 dtype 以对齐双标杆检查

## 目录结构

```text
jagged_flash_attention_basic_kernel/
├── README.md
├── gpu_atk_test/
│   ├── jagged_flash_attention_basic_kernel.py       # GPU 端 Triton kernel 实现
│   └── jagged_flash_attention_basic_kernel_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── jagged_flash_attention_basic_kernel.py       # NPU 端 Triton kernel 实现（autotune）
    ├── jagged_flash_attention_basic_kernel.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_jagged_flash_attention_basic_kernel.py  # ATK 用例生成器
    ├── jagged_flash_attention_basic_kernel_api.py   # NPU 端 ATK API 执行文件
    └── jagged_flash_attention_basic_kernel_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch
from jagged_flash_attention_basic_kernel import jagged_flash_attention_basic_kernel

B, seq_len, D = 2, 32, 64
device = "npu"
q = torch.randn(B * seq_len, D, dtype=torch.float32, device=device)
k = torch.randn(D, B * seq_len, dtype=q.dtype, device=device)
v = torch.randn(B * seq_len, D, dtype=q.dtype, device=device)
offset = torch.tensor([0, seq_len, B * seq_len], dtype=torch.int64, device=device)
o = torch.empty_like(q)
lse = torch.empty((B * seq_len,), dtype=q.dtype, device=device)
grid = (1, B)
jagged_flash_attention_basic_kernel[grid](
    q, k, v, offset, o, lse,
    q.stride(0), q.stride(1), k.stride(0), k.stride(1),
    v.stride(0), v.stride(1), o.stride(0), o.stride(1),
    seq_len, D, 64, False, False,
    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_D=32,
)
print(o.shape, lse.shape)  # torch.Size([64, 64]) torch.Size([64])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成，性能测试用例复用 `npu_atk_test/jagged_flash_attention_basic_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f jagged_flash_attention_basic_kernel.yaml -p generate_jagged_flash_attention_basic_kernel.py
```

生成器约束 q/k/v/offset 的 shape 与 dtype，保证与 kernel 计算口径一致。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c jagged_flash_attention_basic_kernel_performance.json --task accuracy \
    -p jagged_flash_attention_basic_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c jagged_flash_attention_basic_kernel_performance.json --task performance_device \
    -p jagged_flash_attention_basic_kernel_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
