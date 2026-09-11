# _multi_head_jagged_flash_attention_bwd_preprocess_kernel

## 来源

`_multi_head_jagged_flash_attention_bwd_preprocess_kernel` 源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_multi_head_jagged_flash_attention.py>

当前套件只覆盖 Multi-Head Jagged Flash Attention 反向传播的预处理 Kernel，不包含前向、
主反向计算和 autograd 封装。

## 算子功能

Kernel 对前向输出 `o` 与输出梯度 `do` 做逐元素乘法，并沿特征维 D 求和：

```text
delta[h, i] = sum(o[h, i, d] * do[h, i, d] for d in range(D))
```

`o_offset` 将序列位置划分为 B 个 Jagged batch。每个程序实例处理一个 batch、一个
attention head 和一块 M 维位置，只计算该 batch 中不超过 `max_seq_len` 的有效行。

## 输入与输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `o` | `(H, T, D)` | bf16 / fp16 / fp32 | Flash Attention 前向输出 |
| `do` | `(H, T, D)` | 与 `o` 一致 | 输出梯度 |
| `o_offset` | `(B + 1,)` | int64 | B 个 Jagged batch 的起止偏移 |
| `num_heads` | 标量 | int | attention head 数 H |
| `max_seq_len` | 标量 | int | 单个 batch 最大处理长度 |
| `BLOCK_M` | 标量 | int | 每个程序处理的序列行数 |
| `BLOCK_D` | 标量 | int | 特征维块大小，必须不小于 D |
| `delta` | `(H, T)` | 与输入精度对齐 | 每行 `o * do` 在 D 维的求和结果 |

ATK API 只采用随机 `o_offset` 的形状，并根据 T 确定性地生成首项为 0、末项为 T 的
单调 offsets，保证 GPU 与 NPU 使用完全相同的 Jagged 布局。double benchmark 产生
FP64 输入时，API 在执行前转为 FP32，返回前再恢复输出 dtype。

## 关键约束

- `o` 与 `do` 均为三维连续张量，形状及 dtype 必须一致。
- `H = num_heads`，`T = o.shape[1]`，`D = o.shape[2]`。
- `o_offset` 长度为 `B + 1`，首项为 0，末项不超过 T，且单调不减。
- `BLOCK_D >= D`；API 会将不足的值向上调整到 2 的幂。
- grid 为 `(ceil(max_seq_len / BLOCK_M), B * H)`。

## 使用样例

```python
import torch
import triton

from triton_multi_head_jagged_flash_attention import (
    _multi_head_jagged_flash_attention_bwd_preprocess_kernel,
)

H, T, D, B = 4, 128, 32, 2
o = torch.randn((H, T, D), device="cuda")
do = torch.randn_like(o)
o_offset = torch.tensor([0, 64, 128], dtype=torch.int64, device="cuda")
delta = torch.zeros((H, T), dtype=torch.float32, device="cuda")
block_m, block_d, max_seq_len = 32, 32, 64
grid = (triton.cdiv(max_seq_len, block_m), B * H)
_multi_head_jagged_flash_attention_bwd_preprocess_kernel[grid](
    o, o_offset, do, delta,
    o.stride(0), o.stride(1), o.stride(2), delta.stride(0),
    num_heads=H, max_seq_len=max_seq_len, D=D,
    BLOCK_M=block_m, BLOCK_D=block_d,
)
```

## 目录文件

```text
_multi_head_jagged_flash_attention_bwd_preprocess_kernel/
├── README.md
├── gpu_atk_test/
│   ├── triton_multi_head_jagged_flash_attention.py
│   └── multi_head_jagged_flash_attention_bwd_preprocess_kernel_api.py
└── npu_atk_test/
    ├── triton_multi_head_jagged_flash_attention.py
    ├── multi_head_jagged_flash_attention_bwd_preprocess_kernel_api.py
    ├── multi_head_jagged_flash_attention_bwd_preprocess_kernel.yaml
    ├── generate_multi_head_jagged_flash_attention_bwd_preprocess_kernel.py
    └── multi_head_jagged_flash_attention_bwd_preprocess_kernel_performance.json
```

## ATK 测试

在 NPU 端生成精度用例：

```bash
atk case -f multi_head_jagged_flash_attention_bwd_preprocess_kernel.yaml \
    -p generate_multi_head_jagged_flash_attention_bwd_preprocess_kernel.py
```

GPU 端启动服务后，在 NPU 端运行精度或性能任务：

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c all__multi_head_jagged_flash_attention_bwd_preprocess_kernel.json --task accuracy \
    -p multi_head_jagged_flash_attention_bwd_preprocess_kernel_api.py
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c multi_head_jagged_flash_attention_bwd_preprocess_kernel_performance.json \
    --task performance_device -p multi_head_jagged_flash_attention_bwd_preprocess_kernel_api.py
```
