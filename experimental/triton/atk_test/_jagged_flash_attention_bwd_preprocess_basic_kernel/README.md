# `_jagged_flash_attention_bwd_preprocess_basic_kernel` 算子文档

## 来源

`_jagged_flash_attention_bwd_preprocess_basic_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/fe980ab54a6e28818d81c8694b6564e7f804418b/fbgemm_gpu/fbgemm_gpu/sll/triton_sll.py#L1992-L2032>

## 功能说明

`_jagged_flash_attention_bwd_preprocess_basic_kernel` 是 Jagged Flash Attention 反向传播前的预处理算子。它根据 `o_offset` 划分变长序列，并为每个有效 token 计算前向注意力输出 `o` 与其上游梯度 `do` 在特征维度上的点积，生成后续注意力反向计算需要的中间量 `delta`。

设：

- `T` 为所有 Jagged 序列的 token 总数；
- `D` 为每个 token 的特征维度；
- `B = o_offset.numel() - 1` 为序列数量；
- 第 `b` 个序列实际参与计算的长度为

```text
M_b = min(o_offset[b + 1] - o_offset[b], max_seq_len)
```

对于第 `b` 个序列内的第 `m` 个有效 token，令 `t = o_offset[b] + m`，算子计算：

```text
delta[t] = sum(o[t, d] * do[t, d] for d in range(D))
         = dot(o[t, :], do[t, :])
```

该计算是逐 token 的特征维归约，不会在不同 token 或不同 Jagged 序列之间累加。`delta` 通常用于 Flash Attention 的 softmax 反向计算。

## 输入

| 参数 | 说明 |
|------|------|
| `o` | Flash Attention 的前向输出，shape 为 `(T, D)` |
| `do` | `o` 对应的上游梯度，shape 为 `(T, D)`，shape、dtype、device 和内存布局应与 `o` 一致 |
| `o_offset` | Jagged 序列偏移，shape 为 `(B + 1,)`；第 `b` 个序列对应 token 区间 `[o_offset[b], o_offset[b + 1])` |
| `max_seq_len` | 单个序列参与预处理的最大 token 数 |
| `D` | token 的特征维度，应等于 `o.shape[1]` |

`o` 和 `do` 支持 `fp32`、`fp16` 和 `bf16`，`o_offset` 使用 `int64`。

## 输出

`delta`：shape 为 `(T,)`，本目录封装返回的 dtype 与 `o` 相同。对于每个参与计算的 token，`delta[t]` 等于 `o[t, :]` 与 `do[t, :]` 的点积。

当某个序列长度超过 `max_seq_len` 时，该序列中超出前缀范围的位置不会被 kernel 写入。若调用方将 `delta` 初始化为零，这些位置保持为零；否则其内容取决于调用前输出缓冲区中的原值。

## 计算示例

下面使用 NPU 上的 PyTorch 运算表达与算子一致的数学语义，不涉及具体的 Kernel 启动参数：

```python
import torch

device = "npu"
T, D = 7, 32
max_seq_len = 3

o = torch.randn(T, D, dtype=torch.float32, device=device)
do = torch.randn_like(o)

# 两个 Jagged 序列，原始长度分别为 4 和 3
o_offset = torch.tensor([0, 4, 7], dtype=torch.int64, device=device)

delta = torch.zeros(T, dtype=o.dtype, device=device)

for b in range(o_offset.numel() - 1):
    begin = int(o_offset[b].item())
    end = min(int(o_offset[b + 1].item()), begin + max_seq_len)
    delta[begin:end] = torch.sum(o[begin:end] * do[begin:end], dim=1)

print(delta.shape)  # torch.Size([7])

# 第一个序列的第 4 个 token 超过 max_seq_len，未参与计算
print(delta[3])     # tensor(0., device='npu:0')
```

## 关键约束

- `o` 和 `do` 必须是二维 tensor，且 shape、dtype、device 和内存布局一致；通常应传入 contiguous tensor。
- `o.shape[1]` 必须等于 `D`，`delta.shape[0]` 必须至少为 `T`。
- `o_offset` 必须是一维 `int64` tensor，长度为 `B + 1`，满足 `o_offset[0] = 0`、`o_offset[-1] = T`，并且单调不减。
- `max_seq_len` 必须为正数；每个序列仅前 `min(sequence_length, max_seq_len)` 个 token 参与计算。
- 空序列可以用相邻且相等的 offset 表示，该序列不会产生任何有效输出。
- 对于被 `max_seq_len` 截断的位置，kernel 不会写入 `delta`，调用方如需确定值应预先初始化输出缓冲区。
- 算子只生成反向传播的中间量 `delta`，不直接计算 `dq`、`dk` 或 `dv`。

## ATK 测试

以下命令约定 NPU 端已部署 `npu_atk_test/` 目录及对应 YAML、生成器、API 和 JSON 文件。当前工作区中的该算子目录仅包含 `gpu_atk_test/`，如 NPU 侧文件尚未同步，需要先完成部署。

生成及测试命令均在 NPU 端的 `npu_atk_test` 目录下执行：

```bash
cd experimental/triton/atk_test/_jagged_flash_attention_bwd_preprocess_basic_kernel/npu_atk_test
```

### 生成测试数据

```bash
atk case -f _jagged_flash_attention_bwd_preprocess_basic_kernel.yaml \
    -p generate__jagged_flash_attention_bwd_preprocess_basic_kernel.py
```

生成的用例配置文件为 `all__jagged_flash_attention_bwd_preprocess_basic_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__jagged_flash_attention_bwd_preprocess_basic_kernel.json \
    --task accuracy \
    -p _jagged_flash_attention_bwd_preprocess_basic_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 \
    node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all__jagged_flash_attention_bwd_preprocess_basic_kernel.json \
    --task performance_device \
    -p _jagged_flash_attention_bwd_preprocess_basic_kernel_api.py
```
