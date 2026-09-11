# _jagged_dense_flash_attention_bwd_dk_kernel

## 来源

`_jagged_dense_flash_attention_bwd_dk_kernel` 算子源自 FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/v1.5.0-release/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_dense_flash_attention.py>

## 算子功能

`_jagged_dense_flash_attention_bwd_dk_kernel` 用于 **Flash Attention 反向传播** 中计算 **K 的梯度 dk**（dense tensor）。该算子是 `jagged_dense_flash_attention` 反向算子族中独立计算 dk 的 kernel，`dq` / `dv` / `dbias` 由其他 kernel 单独计算。

- 输入侧 Query/Value 为 jagged tensor（变长序列拼接），Key 为 dense tensor（统一最大长度）。
- 输出 `dk` 与 `K` 形状一致 `([B, D, T])`，dense 表示。
- 算法遵循 Flash Attention 反向传播：对 Q/K 做 tile 化的 `tl.dot`，复用 `lse`、`delta` 等前向 / 预处理结果，在 HBM 中只写入 `dk`。
- 依赖前置预处理 `_bwd_preprocess_do_o_dot`，计算 `delta = rowsum(O * do)`，供反向使用。

---

## 输入

`_jagged_dense_flash_attention_bwd_dk_kernel` 算子接受以下输入：

| 参数 | 说明 |
|------|------|
| `Q` | jagged tensor，shape `(sum_B, D)`，`dtype` 支持 fp32 |
| `K` | dense tensor，shape `(B, D, T)`，`dtype` 支持 fp32 |
| `V` | jagged tensor，shape `(sum_B, D)`，`dtype` 支持 fp32 |
| `attn_bias` | dense tensor，shape `(B, N, T)`，加到 `QK^T` 上的偏置 |
| `jagged_offsets` | int64 偏移张量，shape `(B + 1)`，单调递增且首元素为 `0`、末元素为 `sum_B` |
| `Out` | dense tensor，shape `(B, T, D)`，前向输出 |
| `do` | dense tensor，shape `(B, T, D)`，Out 的上游梯度 |
| `lse` | dense tensor，shape `(B, T)`，前向 log-sum-exp |
| `delta` | dense tensor，shape `(B, T)`，`_bwd_preprocess_do_o_dot` 预计算的 `rowsum(O * do)` |
| `dq` | jagged tensor，shape `(sum_B, D)`，反向使用（本 kernel 不写） |
| `dk` | 输出 dense tensor，shape `(B, D, T)`，由本 kernel 写入 |
| `dv` | 反向使用（本 kernel 不计算），调用时传 `None` |
| `dbias` | 反向使用（本 kernel 不计算），调用时传 `None` |
| `max_seq_len` | 每个序列允许的最大长度（`constexpr`），超出会被截断 |
| `BLOCK_T` | K/T 方向 block 大小（`constexpr`，常用 `32`，NPU 端由 autotune 自动选择） |
| `BLOCK_L` | Q/L 方向 block 大小（`constexpr`，常用 `32`，NPU 端由 autotune 自动选择） |
| `BLOCK_D` | D 方向 block 大小（`constexpr`，通常等于 `D`） |
| `allow_tf32` | 是否允许 TF32（`constexpr`） |

> 说明：NPU 端 kernel 使用 `triton.autotune` 包装，由 `_JAGGED_DENSE_FLASH_ATTENTION_BWD_DK_AUTOTUNE_CONFIGS` 列表在 `BLOCK_T ∈ {16, 32, 64, 128}`、`BLOCK_L ∈ {16, 32}`、`num_warps ∈ {1, 2, 4, 8}` 上自动选优，grid 由 `META["BLOCK_T"]` 动态计算。

> 说明：`sum_B = sum(seq_len[i])`，即所有 batch 内序列长度之和；`B`、`T` 来自 `K` 的形状，`N` 来自 `attn_bias` 的中间维。

## 输出

`dk`：dense tensor，shape `(B, D, T)`，`dtype` 与输入张量相同。该 kernel 只产出 `dk`，`dq` / `dv` / `dbias` 由其他 kernel 计算。

---

## 使用样例

### 示例：直接调用 kernel

```python
import torch
import triton
from _jagged_dense_flash_attention_bwd_dk_kernel import (
    _jagged_dense_flash_attention_bwd_dk_kernel,
    _bwd_preprocess_do_o_dot,
)

DEVICE = "npu:0"
B, D, T, N = 4, 16, 64, 8
max_seq_len = 128

# jagged 输入 Q / V / dq 的总 token 数
sum_B = B * (T // 2)

# 随机初始化输入张量
Q      = torch.randn(sum_B, D, dtype=torch.float32, device=DEVICE)
K      = torch.randn(B, D, T,    dtype=torch.float32, device=DEVICE)
V      = torch.randn(sum_B, D, dtype=torch.float32, device=DEVICE)
Out    = torch.randn(B, T, D,    dtype=torch.float32, device=DEVICE)
do     = torch.randn(B, T, D,    dtype=torch.float32, device=DEVICE)
lse    = torch.randn(B, T,       dtype=torch.float32, device=DEVICE)
abias  = torch.randn(B, N, T,    dtype=torch.float32, device=DEVICE)
dq     = torch.zeros(sum_B, D,   dtype=torch.float32, device=DEVICE)

# 构造 jagged_offsets：首元素为 0，末元素为 sum_B
jagged_offsets = torch.tensor(
    [0, sum_B // 4, sum_B // 2, 3 * sum_B // 4, sum_B],
    dtype=torch.int64, device=DEVICE,
)

# 预计算 delta = rowsum(O * do)
delta = torch.empty_like(lse)
BLOCK_T, BLOCK_D = 32, D
num_blocks_t = triton.cdiv(T, BLOCK_T)
_bwd_preprocess_do_o_dot[(num_blocks_t, B)](
    Out, do, delta,
    T,
    Out.stride(0), Out.stride(1), Out.stride(2),
    do.stride(0),  do.stride(1),  do.stride(2),
    BLOCK_T, BLOCK_D,
)

# 调用本 kernel 计算 dk（GPU 端示例：固定 BLOCK_T / BLOCK_L）
dk = torch.zeros_like(K)
BLOCK_T, BLOCK_L, BLOCK_D = 32, 32, D
num_blocks_t = triton.cdiv(T, BLOCK_T)
_jagged_dense_flash_attention_bwd_dk_kernel[(num_blocks_t, B)](
    Q, K, V, abias, jagged_offsets,
    Out, do, lse, delta, dq, dk,
    None,                                  # dv: 本 kernel 不计算
    None,                                  # dbias: 本 kernel 不计算
    max_seq_len,
    Q.stride(0), Q.stride(1),
    K.stride(0), K.stride(1), K.stride(2),
    V.stride(0), V.stride(1),
    abias.stride(0), abias.stride(1), abias.stride(2),
    Out.stride(0), Out.stride(1), Out.stride(2),
    dk.stride(0),  dk.stride(1),  dk.stride(2),
    do.stride(0),  do.stride(1),  do.stride(2),
    D, T,
    BLOCK_T, BLOCK_L, BLOCK_D,
    allow_tf32=False,
)

# NPU 端示例：kernel 已用 triton.autotune 包装，grid 由 META["BLOCK_T"] 动态计算
def _grid(META):
    return (triton.cdiv(T, META["BLOCK_T"]), B)

_jagged_dense_flash_attention_bwd_dk_kernel[_grid](
    q_ptr=Q, k_ptr=K, v_ptr=V, ab_ptr=abias,
    jagged_offsets_ptr=jagged_offsets,
    out_ptr=Out, do_ptr=do, lse_ptr=lse, delta_ptr=delta,
    dq_ptr=dq, dk_ptr=dk,
    dv_ptr=None, dbias_ptr=None,
    max_seq_len=max_seq_len,
    stride_ql=Q.stride(0), stride_qd=Q.stride(1),
    stride_kb=K.stride(0), stride_kd=K.stride(1), stride_kt=K.stride(2),
    stride_vl=V.stride(0), stride_vd=V.stride(1),
    stride_ab_b=abias.stride(0), stride_ab_l=abias.stride(1), stride_ab_t=abias.stride(2),
    stride_ob=Out.stride(0), stride_ot=Out.stride(1), stride_od=Out.stride(2),
    stride_dk_b=dk.stride(0), stride_dk_d=dk.stride(1), stride_dk_t=dk.stride(2),
    stride_do_b=do.stride(0), stride_do_t=do.stride(1), stride_do_d=do.stride(2),
    D=D, T=T,
    BLOCK_D=D,
    allow_tf32=False,
)

torch.npu.synchronize(DEVICE)
print("dk.shape:", tuple(dk.shape))        # (B, D, T)
```

---

## 关键约束

- `D` 必须是 **2 的幂次方**，且 `D >= 16`（Triton `tl.dot` 对 K 维度的最小要求）。
- `K` 形状为 `[B, D, T]`；`Q`、`V`、`dq` 形状为 `[sum_B, D]`；`Out`、`do` 形状为 `[B, T, D]`。
- `lse`、`delta` 形状为 `[B, T]`；`attn_bias` 形状为 `[B, N, T]`。
- `jagged_offsets` 形状为 `[B + 1]`，单调递增，首元素为 `0`，末元素为 `sum_B`。
- 每个序列实际长度 `seqlen = jagged_offsets[i+1] - jagged_offsets[i]`，kernel 内部以 `min(seqlen, max_seq_len)` 截断。
- `max_seq_len >= T`，否则所有序列都会被截断到 `max_seq_len`，结果退化。
- `delta` 由前置 kernel `_bwd_preprocess_do_o_dot(Out, do) -> [B, T]` 计算得到，调用本 kernel 前需先准备好。
- 本 kernel **不计算** `dq` / `dv` / `dbias`，对应位置需传 `None`。

---

## 目录文件

```text
_jagged_dense_flash_attention_bwd_dk_kernel/
├── README.md
├── gpu_atk_test/
│   ├── _jagged_dense_flash_attention_bwd_dk_kernel.py                # 原版算子 kernel 实现
│   └── _jagged_dense_flash_attention_bwd_dk_kernel_api.py            # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── _jagged_dense_flash_attention_bwd_dk_kernel.py                # NPU 适配优化后算子 kernel 实现
    ├── _jagged_dense_flash_attention_bwd_dk_kernel_api.py            # NPU 端 ATK 自定义 API 封装
    ├── _jagged_dense_flash_attention_bwd_dk_kernel.yaml              # ATK 算子用例参数
    ├── generate_jagged_dense_flash_attention_bwd_dk_kernel.py        # 限制参数生成脚本
    └── _jagged_dense_flash_attention_bwd_dk_kernel_performance.json   # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/_jagged_dense_flash_attention_bwd_dk_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成精度测试数据

```bash
atk case -f _jagged_dense_flash_attention_bwd_dk_kernel.yaml -p generate_jagged_dense_flash_attention_bwd_dk_kernel.py
```

生成的精度测试用例位于 `result/_jagged_dense_flash_attention_bwd_dk_kernel/json/` 下，文件名为 `all__jagged_dense_flash_attention_bwd_dk_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c all__jagged_dense_flash_attention_bwd_dk_kernel.json --task accuracy \
    -p _jagged_dense_flash_attention_bwd_dk_kernel_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c _jagged_dense_flash_attention_bwd_dk_kernel_performance.json --task performance_device \
    -p _jagged_dense_flash_attention_bwd_dk_kernel_api.py
```

注：如需保存性能对比数据则需要在指令后加 --save_data profile
