# _bwd_preprocess_do_o_dot

## 来源

`_bwd_preprocess_do_o_dot` 算子源自 Dao-AILab FlashAttention 的 Triton 实现：

<https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py>

原始算子按 batch、head 和序列位置计算输出与输出梯度的点积。本目录中的 ATK
实现将 batch 与 head 统一表示为 `B`，输入形状简化为 `[B, T, D]`。

## 功能说明

`_bwd_preprocess_do_o_dot` 是 Flash Attention 反向传播的预处理算子。它将前向输出
`Out` 与输出梯度 `do` 逐元素相乘，并沿最后一维 `D` 求和：

```text
delta[b, t] = sum(Out[b, t, d] * do[b, t, d] for d in range(D))
```

计算结果 `delta` 会在后续反向计算中用于 softmax 梯度修正。

### 输入

- `Out` `(B, T, D)` — Flash Attention 前向输出
- `do` `(B, T, D)` — 反向传播传入的输出梯度

### 输出

- `delta` `(B, T)` — `Out` 与 `do` 在特征维度上的逐位置点积

### 约束

- `Out` 与 `do` 必须是形状相同的三维 tensor
- `Out` 与 `do` 必须使用相同 dtype
- 支持 `fp32`、`fp16`、`bf16`
- `B > 0`、`T > 0`、`D > 0`
- 用例生成器会将 `D` 调整为不小于原值的最近 2 的幂

## 目录结构

```text
experimental/triton/atk_test/_bwd_preprocess_do_o_dot/
├── README.md
├── gpu_atk_test/
│   ├── _bwd_preprocess_do_o_dot_kernel.py       # GPU(CUDA) 端 Triton kernel
│   └── _bwd_preprocess_do_o_dot_api.py          # GPU(CUDA) 端 ATK API
└── npu_atk_test/
    ├── _bwd_preprocess_do_o_dot_kernel.py       # NPU(Triton) kernel
    ├── _bwd_preprocess_do_o_dot.yaml            # ATK 用例设计
    ├── generate__bwd_preprocess_do_o_dot.py     # ATK 用例生成器
    ├── _bwd_preprocess_do_o_dot_api.py          # NPU(Triton) 端 ATK API
    └── _bwd_preprocess_do_o_dot_performance.json # 150 个 ATK 用例
```

## 调用示例

以下示例使用 NPU kernel；`BLOCK_T` 由 autotune 选择：

```python
import torch
import triton

from _bwd_preprocess_do_o_dot_kernel import _bwd_preprocess_do_o_dot

B, T, D = 8, 128, 64
device = "npu"
out = torch.randn((B, T, D), dtype=torch.float32, device=device)
do = torch.randn_like(out)
delta = torch.empty((B, T), dtype=out.dtype, device=device)

def grid(meta):
    return (triton.cdiv(T, meta["BLOCK_T"]), B)

_bwd_preprocess_do_o_dot[grid](
    out,
    do,
    delta,
    T,
    out.stride(0),
    out.stride(1),
    out.stride(2),
    do.stride(0),
    do.stride(1),
    do.stride(2),
    BLOCK_D=D,
)
print(delta.shape)  # torch.Size([8, 128])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度和性能测试。

### 远端 GPU 服务器启动监听

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

### 生成测试数据

在 NPU 端的 `npu_atk_test/` 目录执行：

```bash
atk case -f _bwd_preprocess_do_o_dot.yaml -p generate__bwd_preprocess_do_o_dot.py
```

生成器会统一 `Out` 与 `do` 的 shape 和 dtype，并将 `D` 调整为 2 的幂。

### 精度对比

```bash
atk node --backend npu --devices 0 task -c _bwd_preprocess_do_o_dot_performance.json --task accuracy -p _bwd_preprocess_do_o_dot_api.py
```

如需保存精度数据，在命令末尾增加 `--save_data profile`。

### 性能对比

```bash
atk node --backend npu --devices 0 task -c _bwd_preprocess_do_o_dot_performance.json --task performance_device -p _bwd_preprocess_do_o_dot_api.py
```

如需保存性能数据，在命令末尾增加 `--save_data profile`。
