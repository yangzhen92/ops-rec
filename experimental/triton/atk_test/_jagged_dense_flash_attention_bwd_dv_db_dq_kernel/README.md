# _jagged_dense_flash_attention_bwd_dv_db_dq_kernel

## 来源

`_jagged_dense_flash_attention_bwd_dv_db_dq_kernel` 算子源自 FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/v1.5.0-release/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_dense_flash_attention.py>

## 算子功能

`_jagged_dense_flash_attention_bwd_dv_db_dq_kernel` 是 **jagged + dense Flash Attention 反向传播** 算子，用于一次性计算 Q / V / attention bias 三者的梯度 `dq`、`dv`、`dbias`，其中：

- **Q、V** 是 **jagged tensor**（不同 batch 序列长度不同，按扁平存储 + offsets 寻址），形状为 `[sum_B, D]`。
- **K、Out、do、lse、attn_bias** 是 **dense tensor**（K 在所有 batch 上共享同一段 key/value 序列），其中 K 形状为 `[B, D, T]`，attn_bias 形状为 `[B, N, T]`。
- 仅计算 `dq`、`dv`、`dbias`；**`dk` 由独立的 kernel 计算**（不在本算子范围内）。

整个算子由两个 Triton kernel 组成：

1. `_bwd_preprocess_do_o_dot`：按 batch 计算 `delta = sum(o * do, axis=-1)`，作为反向 softmax 的中间量。
2. `_jagged_dense_flash_attention_bwd_dv_db_dq_kernel`：核心 kernel，对 `(B, max_seq_len)` 二维 grid 启动，按 jagged offsets 切片 Q / V，按 dense K 做 attention 反向，对每个 `(BLOCK_L, BLOCK_T)` 分块依次累加 `dq`、`dv`、`dbias`。

### 反向算法概览

对每个 batch `b` 与 Q 行块 `l`：

```text
# 1) 预计算 delta
delta[t] = sum_d(Out[b, t, d] * do[b, t, d])

# 2) 沿 T 维累加 (tl.load BLOCK_T * stride 步进)
for t in range(0, T, BLOCK_T):
    k   = K[b, :, t:t+BLOCK_T]                # [D, BLOCK_T]
    qk  = Q[l] @ k + attn_bias[b, l, t]       # [BLOCK_L, BLOCK_T]
    qk  = mask(qk, valid_l, valid_t, -inf)
    p   = exp(qk - lse[b, t])                 # softmax
    do  = do[b, t:t+BLOCK_T, :]               # [BLOCK_T, D]

    dv[l] += p @ do                           # gemm: [BLOCK_L, D]
    dp    = (do @ V[l].T).T                   # [BLOCK_L, BLOCK_T]
    ds    = p * (dp - delta[b, t])            # softmax 反向
    dbias[b, l, t] = ds                       # bias 梯度
    dq[l] += ds @ k.T                         # gemm: [BLOCK_L, D]
```

---

## 输入

`_jagged_dense_flash_attention_bwd_dv_db_dq_kernel` 算子接受以下输入：

| 参数 | 说明 |
|------|------|
| `Q` | Q 的 jagged 扁平值，shape `(sum_B, D)`，`dtype = fp32` |
| `K` | 共享的 K，dense tensor，shape `(B, D, T)`，`dtype = fp32` |
| `V` | V 的 jagged 扁平值，shape `(sum_B, D)`，`dtype = fp32` |
| `Out` | 前向输出，dense tensor，shape `(B, T, D)`，用于预计算 `delta`，`dtype = fp32` |
| `lse` | 前向 `log-sum-exp`，dense tensor，shape `(B, T)`，`dtype = fp32` |
| `do` | 输出梯度，dense tensor，shape `(B, T, D)`，`dtype = fp32` |
| `attn_bias` | attention bias，dense tensor，shape `(B, N, T)`，`dtype = fp32` |
| `jagged_offsets` | Q / V 的分段偏移，shape `(B + 1)`，`int64`，首元素为 `0`、末元素为 `sum_B` |
| `max_seq_len` | Q 单行最大长度（`int`），超过该长度的序列会被截断 |
| `allow_tf32` | 是否允许 `tl.dot` 使用 TF32（`attr_bool`） |
| `T` | 关键常量（`constexpr`），通常取 `max_seq_len`，用于 tile 循环上界 |
| `BLOCK_T` | K / do 在 T 维的 block 大小（`constexpr`，常用 `32`） |
| `BLOCK_L` | Q / V / dq / dv 在 L 维的 block 大小（`constexpr`，常用 `32`） |
| `BLOCK_D` | hidden 维 block 大小（`constexpr`），通常取 `D` 本身 |
| 各 `stride_*` | 上述每个张量的 batch / length / d / t 维 stride（由调用方计算后传入） |

> 注意：`Out` 与 `do` 走的是 dense 的 `(B, T, D)` 布局，而 Q / V / dq / dv 走的是 jagged 的 `(sum_B, D)` 布局。两者通过 `jagged_offsets[pid_b]` / `jagged_offsets[pid_b + 1]` 切片关联。

## 输出

- `dq`：shape `(sum_B, D)`，dtype 与 `Q` 一致，jagged 布局，**仅在 `offs_l_curr < seqlen` 范围内有效**。
- `dv`：shape `(sum_B, D)`，dtype 与 `V` 一致，jagged 布局，写入规则同上。
- `dbias`：shape `(B, N, T)`，dtype 与 `attn_bias` 一致，dense 布局，仅在 `valid_l & valid_t` 掩码内写入。

`dk` 不由本 kernel 计算，调用方需自行补齐。

---

## 使用样例

### 示例：直接调用 kernel

```python
import torch
import triton
from _jagged_dense_flash_attention_bwd_dv_db_dq_kernel import (
    _jagged_dense_flash_attention_bwd_dv_db_dq_kernel,
    _bwd_preprocess_do_o_dot,
)

DEVICE = "npu:0"
B, N, D, T = 4, 8, 64, 32
MAX_SEQ_LEN = T

seq_lens = [12, 8, 16, 10]
sum_B = sum(seq_lens)
jagged_offsets = torch.tensor(
    [0] + seq_lens, dtype=torch.int64, device=DEVICE
).cumsum_(dim=0)
jagged_offsets[0] = 0
jagged_offsets[-1] = sum_B

Q   = torch.randn(sum_B, D, dtype=torch.float32, device=DEVICE)
V   = torch.randn(sum_B, D, dtype=torch.float32, device=DEVICE)
K   = torch.randn(B,   D, T, dtype=torch.float32, device=DEVICE)
Out = torch.randn(B,   T, D,   dtype=torch.float32, device=DEVICE)
do  = torch.randn(B,   T, D,   dtype=torch.float32, device=DEVICE)
lse = torch.randn(B,   T,     dtype=torch.float32, device=DEVICE)
attn_bias = torch.randn(B, N, T, dtype=torch.float32, device=DEVICE)

dq    = torch.zeros_like(Q)
dv    = torch.zeros_like(V)
dbias = torch.zeros_like(attn_bias)
delta = torch.empty_like(lse)

BLOCK_T, BLOCK_L, BLOCK_D = 32, 32, D

# 1) 预计算 delta = sum(o * do)
num_blocks_t = triton.cdiv(T, BLOCK_T)
_bwd_preprocess_do_o_dot[(num_blocks_t, B)](
    Out, do, delta, T,
    Out.stride(0), Out.stride(1), Out.stride(2),
    do.stride(0),  do.stride(1),  do.stride(2),
    BLOCK_T, BLOCK_D,
)
torch.npu.synchronize(DEVICE)

# 2) 计算 dq / dv / dbias
# 注意：kernel 已加 @triton.autotune，BLOCK_T / BLOCK_L 由 autotuner 从 configs 注入，
# 调用时不能显式传；grid 需从 META 读取 BLOCK_L（该 kernel 按 BLOCK_L 拆 L 维）
def _grid(META):
    return (triton.cdiv(MAX_SEQ_LEN, META["BLOCK_L"]), B)

_jagged_dense_flash_attention_bwd_dv_db_dq_kernel[_grid](
    q_ptr=Q, k_ptr=K, v_ptr=V, ab_ptr=attn_bias,
    jagged_offsets_ptr=jagged_offsets,
    out_ptr=Out, do_ptr=do, lse_ptr=lse, delta_ptr=delta,
    dq_ptr=dq, dk_ptr=None, dv_ptr=dv, dbias_ptr=dbias,
    max_seq_len=MAX_SEQ_LEN,
    stride_ql=Q.stride(0),            stride_qd=Q.stride(1),
    stride_kb=K.stride(0),            stride_kd=K.stride(1),        stride_kt=K.stride(2),
    stride_vl=V.stride(0),            stride_vd=V.stride(1),
    stride_ab_b=attn_bias.stride(0),  stride_ab_l=attn_bias.stride(1), stride_ab_t=attn_bias.stride(2),
    stride_ob=Out.stride(0),          stride_ot=Out.stride(1),      stride_od=Out.stride(2),
    stride_dq_l=dq.stride(0),         stride_dq_d=dq.stride(1),
    stride_dv_l=dv.stride(0),         stride_dv_d=dv.stride(1),
    stride_db_b=dbias.stride(0),      stride_db_l=dbias.stride(1),  stride_db_t=dbias.stride(2),
    stride_do_b=do.stride(0),         stride_do_t=do.stride(1),     stride_do_d=do.stride(2),
    T=T,
    BLOCK_D=BLOCK_D,
    allow_tf32=True,
)

torch.npu.synchronize(DEVICE)
# dq: torch.Size([46, 64]) torch.float32
# dv: torch.Size([46, 64]) torch.float32
# dbias: torch.Size([4, 8, 32]) torch.float32
print("dq:",    dq.shape,    dq.dtype)
print("dv:",    dv.shape,    dv.dtype)
print("dbias:", dbias.shape, dbias.dtype)
```

---

## 关键约束

- `jagged_offsets[0] == 0` 且 `jagged_offsets[-1] == sum_B = Q.shape[0] = V.shape[0]`。
- `jagged_offsets` 单调递增；当前 batch 的真实序列长度 `seqlen = jagged_offsets[pid_b + 1] - jagged_offsets[pid_b]`，超过 `max_seq_len` 的部分会被截断。
- `D`（hidden 维）必须是 **2 的幂次方且 `>= 16`**：Triton `tl.arange(0, BLOCK_D)` 要求范围为 2 的幂次方，`tl.dot` 要求 inner dim `>= 16`。
- **计算 dtype 强制 fp32**：kernel 内 `dq / dv / qk / dp` 累加器显式声明为 `tl.float32`，`tl.dot` 也未指定 `out_dtype`，故所有张量按 fp32 输入与计算。
- `max_seq_len >= T`：避免 T 维 tile 循环上界小于实际 key/value 长度。
- `BLOCK_D` 一般取 `D` 本身，以避免额外的 d 维切分。
- Q / V / dq / dv 在 `(sum_B, D)` 上需保证连续；K / Out / do / attn_bias 在对应 dense 布局上需保证连续。
- `Out` 与 `do` 是 dense `(B, T, D)`，与 jagged Q / V 通过 batch 维索引对齐（**不是行对齐**）。

---

## 目录文件

```text
_jagged_dense_flash_attention_bwd_dv_db_dq_kernel/
├── README.md
├── gpu_atk_test/
│   ├── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel.py        # 原版算子 kernel 实现（含 _bwd_preprocess_do_o_dot）
│   └── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_api.py    # GPU 端 ATK 自定义 API 封装
└── npu_atk_test/
    ├── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel.py        # NPU 适配优化后算子 kernel 实现
    ├── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_api.py    # NPU 端 ATK 自定义 API 封装
    ├── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel.yaml       # ATK 算子用例参数
    ├── generate_jagged_dense_flash_attention_bwd_dv_db_dq_kernel.py # 限制参数生成脚本
    └── _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_performance.json  # 性能测试样例数据
```

---

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/_jagged_dense_flash_attention_bwd_dv_db_dq_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成精度测试数据

```bash
atk case -f _jagged_dense_flash_attention_bwd_dv_db_dq_kernel.yaml \
    -p generate_jagged_dense_flash_attention_bwd_dv_db_dq_kernel.py
```

生成的精度测试用例位于 `result/_jagged_dense_flash_attention_bwd_dv_db_dq_kernel/json/` 下，文件名为 `all__jagged_dense_flash_attention_bwd_dv_db_dq_kernel.json`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c all__jagged_dense_flash_attention_bwd_dv_db_dq_kernel.json --task accuracy \
    -p _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_api.py
```

### 性能测试

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端监听端口} --devices 0 \
    task -c _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_performance.json --task performance_device \
    -p _jagged_dense_flash_attention_bwd_dv_db_dq_kernel_api.py
```
