# jagged_flash_attention_basic_kernel

## 来源

`jagged_flash_attention_basic_kernel` 算子源自 PyTorch FBGEMM 代码仓：<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_flash_attention_basic.py>
## 功能说明
Jagged Flash Attention 前向计算：对每个 batch 的变长序列执行 `o, lse = flash_attention(q, k, v, offset, max_seq_len)`，采用 FlashAttention 分块在线 softmax。
- 输入：`q` `(total_seq, D)`；`k` `(D, total_seq)`（转置形式）；`v` `(total_seq, D)`；`offset` `(B+1,)` int64；`max_seq_len`；`use_mask`（因果 mask）；`allow_tf32`；`BLOCK_SIZE_M/N/D`
- 输出：`o` `(total_seq, D)`；`lse` `(total_seq,)`
- 约束：q/k/v dtype 一致（`fp32`/`fp16`/`bf16`）；`D` 取 2 的幂次方；`total_seq ≥ B`（B=4~10）；offset 首 0 尾 `total_seq`
- 调优：GPU 端固定 32×32×32；NPU 端 `@triton.autotune` 按 `[max_seq_len, D]` 选 BLOCK（16/32 档）；NPU 输入强转 fp32 编译、输出转回原 dtype
## 目录结构

```text
jagged_flash_attention_basic_kernel/
├── README.md
├── gpu_atk_test/      # jagged_flash_attention_basic_kernel.py + jagged_flash_attention_basic_kernel_api.py
└── npu_atk_test/      # kernel（autotune）+ .yaml + generate_*.py + _api.py + _performance.json
```
## 调用示例

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
    q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1),
    seq_len, D, 64, False, False,
    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_D=32,
)
print(o.shape, lse.shape)  # torch.Size([64, 64]) torch.Size([64])
```
## ATK 测试
- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端；GPU 端起 ATK server，NPU 端作为客户端拉起精度/性能测试
- 精度用例在 NPU 端生成（生成器约束 q/k/v/offset 的 shape 与 dtype）；性能用例复用 `npu_atk_test/jagged_flash_attention_basic_kernel_performance.json`

```bash
atk case -f jagged_flash_attention_basic_kernel.yaml -p generate_jagged_flash_attention_basic_kernel.py
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_jagged_flash_attention_basic_kernel.json --task accuracy -p jagged_flash_attention_basic_kernel_api.py
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c jagged_flash_attention_basic_kernel_performance.json --task performance_device -p jagged_flash_attention_basic_kernel_api.py
```
