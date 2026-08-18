# hstu_v1 / hstu_forward_v1

## 说明

本算子仅支持 NPU 调用，由 RecSDK `cust_op` 工程迁移而来（源码映射见文末「迁移说明」）。

`hstu_v1` 为 HSTU 子组（转发层），内部按版本收敛多算子 vendor：当前仅包含 `hstu_forward_v1`（HSTU 前向融合注意力，HstuDenseForward / HstuJaggedForward / HstuPagedForward 三个 OpDef）。

## 产品支持情况

| 硬件型号             | 是否支持 |
| -------------------- | -------- |
| Atlas A2 训练系列产品 | 是       |
| Atlas A3 训练系列产品 | 是       |
| Atlas A5 训练系列产品 | 是       |

## 功能

推荐场景下，使用 HSTU 融合算子实现注意力机制。

**GQA 支持**：本算子支持 Grouped Query Attention (GQA)，允许 K/V 的头数小于 Q 的头数，多个 Q 头可以共享同一个 K/V 头，从而减少 KV 缓存内存占用并提升推理性能。

**dim 不等支持**：本算子支持 qk 的 head_dim 与 v_dim 不相等的场景。

**jagged 双模式**：jagged 布局支持 equal 与 delta 两种模式。`hstu_jagged.equal` 中 q 与 k/v 共用同一个 `seq_offset`（每段内 q/k 长度一致）；`hstu_jagged.delta` 中 q 与 k/v 使用不同的 `seq_offset` / `seq_offset_k`，允许 q 段与 k 段长度不同（增量推理 / prefix-caching 场景）。

**paged 支持**：`hstu_paged` 面向 KV cache 分页解码场景，历史 KV 按 `page_size` 分页存放于 `kv_cache`，通过 `page_offsets` / `page_ids` / `last_page_len` 描述分页布局，`num_target` + `target_group_size` 描述当前批次新生成的（目标）token。

**FP8 支持**：本算子仅在 A5 上支持 data_type 为 fp8_e4m3fn 的 qkv 输入，算子输出为 float16。

## 算子实现原理

### 计算公式

$$
HSTU(q, k, v, mask, bias, siluScale) = (Silu(qk^{T} + bias) \times siluScale \times mask)v
$$

### 数据格式

输入参数 q, k, v 数据格式为 normal、jagged 或 paged：

* **normal（dense）格式**：shape 为 `[B, S, N, D]` 的 4 维数据
* **jagged 格式**：shape 为 `[s_b, N, D]` 的 3 维数据，配合 `seq_offset_q`/`seq_offset_k` 使用（equal 模式二者相同，delta 模式可不同）
* **paged 格式**：3 维数据配合 `kv_cache` 分页存储使用（KV cache 分页解码场景）

### 计算逻辑

```python
def hstu_dense_forward(q_np, k_np, v_np, rel_attn_bias_np, invalid_attn_mask_np):
    q = torch.nn.Parameter(torch.Tensor(q_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu())
    k = torch.nn.Parameter(torch.Tensor(k_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu())
    v = torch.nn.Parameter(torch.Tensor(v_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu())
    real_attn_bias = torch.nn.Parameter(torch.Tensor(rel_attn_bias_np).to(dataType).npu())
    invalid_attn_mask = torch.Tensor(invalid_attn_mask_np).to(dataType).npu()

    qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)
    qk_attn = qk_attn + rel_attn_bias
    qk_attn = F.silu(qk_attn) / max_seq_len
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v).reshape(batch_size, seq_len, num_heads * attention_dim)
    return attn_output
```

## 算子输入与输出

### 公共参数（三种 layout 通用）

| 名称            | 输入/输出 | 数据类型 | 说明 |
|-----------------|-------|---------|------|
| q               | 输入  | Tensor[float32/float16/bfloat16]，A5 另支持 fp8_e4m3fn | `[B, S, N_q, D_q]`（normal）或 `[s_b, N_q, D_q]`（jagged/paged） |
| k               | 输入  | 同 q（GQA 时头数可小于 Q） | `[B, S, N_k, D_q]` 或 `[s_b, N_k, D_q]` |
| v               | 输入  | 同 q（dim 可与 qk 不等） | `[B, S, N_k, D_v]` 或 `[s_b, N_k, D_v]` |
| mask            | 输入  | 同 q（可选） | `[B, N, S, S]`，mask_type=3 时必传 |
| attn_bias       | 输入  | 同 q（可选） | `[B, N, S, S]` 相对位置偏置 |
| mask_type       | 输入  | int | 0：内置下三角；1：上三角（暂不支持）；2：不使用；3：自定义 mask |
| max_seq_len_q   | 输入  | int | Q 序列最大长度，`[1, 20480]` |
| max_seq_len_k   | 输入  | int | K 序列最大长度，`[1, 20480]` |
| silu_scale      | 输入  | float | 默认 `1/max_seq_len` |
| deterministic   | 输入  | bool | 确定性计算开关，默认 false |
| attn_output     | 输出  | Tensor | `[B, S, N_q, D_v]` 或 `[s_b, N_q, D_v]`，A5 的 FP8 输入输出为 fp16 |

### layout 专属参数

| layout   | 专属参数 |
|----------|----------|
| normal   | 无（4 维 q/k/v 直接使用） |
| jagged   | seq_offset_q、seq_offset_k（`[B+1]`，int32/int64）、num_context、num_target（可选） |
| paged    | seq_offset_q/k/t、kv_cache（`[num_pages, 2, page_size, N, D]`）、page_offsets、page_ids、last_page_len、num_target、target_group_size |

> 详细参数说明见 [docs/torch_hstu_forward.md](docs/torch_hstu_forward.md)。

## 调用方式

四种接口均通过 `ops_rec.ascendc.attention.*` 调用（先安装 whl 并 `import ops_rec`），参数名与 torch schema 一一对应，未给出的可选参数使用默认值；q/k/v 及 mask、attn_bias、seq_offset 等 Tensor 参数需位于 NPU 设备上（`.npu()`）。

### hstu_dense（normal 格式）

```python
import torch
import ops_rec

B, S, N, D = 2, 256, 8, 64
q = torch.randn(B, S, N, D, dtype=torch.float16).npu()
k = torch.randn(B, S, N, D, dtype=torch.float16).npu()
v = torch.randn(B, S, N, D, dtype=torch.float16).npu()
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
print(y.shape)  # torch.Size([2, 256, 8, 64])
```

### hstu_jagged.equal（jagged 格式，q 与 k/v 共用 seq_offset）

```python
s_b, N, D = 512, 8, 64
q = torch.randn(s_b, N, D, dtype=torch.float16).npu()
k = torch.randn(s_b, N, D, dtype=torch.float16).npu()
v = torch.randn(s_b, N, D, dtype=torch.float16).npu()
seq_offset = torch.tensor([0, 256, 512], dtype=torch.int32).npu()  # [B+1]，两段各 256 行
y = ops_rec.ascendc.attention.hstu_jagged.equal(
    q, k, v, None, None, 0, 256, 1.0 / 256, seq_offset)
print(y.shape)  # torch.Size([512, 8, 64])
```

### hstu_jagged.delta（jagged 格式，q 与 k/v 使用不同 seq_offset）

```python
# q 段长 [4, 6]，k/v 段长 [6, 8]，段长可以不同
N, D = 4, 32
q = torch.randn(10, N, D, dtype=torch.float16).npu()
k = torch.randn(14, N, D, dtype=torch.float16).npu()
v = torch.randn(14, N, D, dtype=torch.float16).npu()
seq_offset = torch.tensor([0, 4, 10], dtype=torch.int32).npu()
seq_offset_k = torch.tensor([0, 6, 14], dtype=torch.int32).npu()
y = ops_rec.ascendc.attention.hstu_jagged.delta(
    q, k, v, None, None, 0, 6, 8, 1.0 / 6, seq_offset, seq_offset_k)
print(y.shape)  # torch.Size([10, 4, 32])
```

### hstu_paged（paged 格式，kv_cache 分页解码）

```python
y = ops_rec.ascendc.attention.hstu_paged(
    q, k, v, kv_cache,
    mask=None, attn_bias=None, mask_type=0,
    max_seq_len=max_seq_len, max_seq_len_k=max_seq_len_k, silu_scale=1.0 / max_seq_len,
    seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
    page_offsets=page_offsets, page_ids=page_ids, last_page_len=last_page_len,
    num_target=num_target, target_group_size=1)
```

kv_cache 与分页元数据（page_offsets / page_ids / last_page_len）的完整构造见下节「特性用法示例」的 paged decode 示例。

## 特性用法示例

以下每个特性均为完整可运行的最小示例（含必要的变量构造，参数名与真实 schema 一致）。

### GQA（N_q=8，N_k=2）

```python
B, S = 2, 128
q = torch.randn(B, S, 8, 64, dtype=torch.float16).npu()   # N_q=8
k = torch.randn(B, S, 2, 64, dtype=torch.float16).npu()   # N_k=2（GQA，每 4 个 Q 头共享 1 个 K/V 头）
v = torch.randn(B, S, 2, 64, dtype=torch.float16).npu()   # N_v = N_k
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
print(y.shape)  # torch.Size([2, 128, 8, 64])
```

约束：`N_q % N_k == 0` 且 `N_k == N_v`。

### qk 与 v 的 dim 不等

```python
B, S = 2, 128
q = torch.randn(B, S, 8, 64, dtype=torch.float16).npu()    # head_dim_qk=64
k = torch.randn(B, S, 8, 64, dtype=torch.float16).npu()
v = torch.randn(B, S, 8, 128, dtype=torch.float16).npu()   # head_dim_v=128（需为 16 的倍数）
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
print(y.shape)  # torch.Size([2, 128, 8, 128])，输出 dim 与 v 一致
```

### 自定义 mask（mask_type=3 + mask 传入）

mask_type=1（上三角）暂不支持；mask_type=3 时 mask 必须传入。

```python
B, S, N = 2, 128, 8
q = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
k = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
v = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
mask = torch.tril(torch.ones(S, S, dtype=torch.float16)).expand(B, N, S, S).npu()  # 下三角 0/1
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask, None, 3, S, 1.0 / S)
```

### attn_bias 相对位置偏置

```python
B, S, N = 2, 128, 8
q = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
k = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
v = torch.randn(B, S, N, 64, dtype=torch.float16).npu()
attn_bias = torch.randn(B, N, S, S, dtype=torch.float16).npu()  # [B, N, S, S] 相对位置偏置
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, attn_bias, 0, S, 1.0 / S)
```

### silu_scale 自定义

`silu_scale` 传 0.0（或默认值）时自动取 `1/max_seq_len`；需要其他缩放值（如按温度缩放）时显式传入：

```python
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, silu_scale=0.05)
```

### deterministic 确定性开关

仅 jagged / paged 接口带 `deterministic` 参数（dense 接口无此参数）：

```python
y = ops_rec.ascendc.attention.hstu_jagged.equal(
    q, k, v, None, None, 0, S, 1.0 / S, seq_offset, deterministic=True)
```

### jagged 的 num_context / num_target（可选参数）

`num_context` / `num_target` 用于标记每个 batch 的上下文（context）与目标（target）token 数量，未传入时默认全 0：

```python
num_context = torch.tensor([256, 256], dtype=torch.int32).npu()
num_target = torch.tensor([0, 0], dtype=torch.int32).npu()
y = ops_rec.ascendc.attention.hstu_jagged.equal(
    q, k, v, None, None, 0, 256, 1.0 / 256, seq_offset,
    num_context=num_context, num_target=num_target)
```

### paged 的 target_group_size + num_target（decode 场景）

decode 场景：每个 batch 的历史 KV 已按 page_size 分页存入 kv_cache，`num_target` 描述本次新生成的 token 数（定义时 `target_group_size` 必须 > 0）：

```python
N, D, page_size, batch = 4, 32, 32, 2   # 注意：page_size 必须 ∈ {32, 64, 128, 256}（tiling 层约束）
new_hist_lens = [32, 48]   # 每个 batch 的历史 token 数（已存入 kv_cache），覆盖恰好满页(32)与末页不足(48 % 32 = 16)
num_targets = [4, 6]       # 每个 batch 新生成的 token 数
seq_lens = [h + t for h, t in zip(new_hist_lens, num_targets)]
total, max_seq_len = sum(seq_lens), max(seq_lens)

q = torch.randn(total, N, D, dtype=torch.float16).npu()
k = torch.randn(total, N, D, dtype=torch.float16).npu()
v = torch.randn(total, N, D, dtype=torch.float16).npu()

seq_offset = torch.tensor([0] + [sum(seq_lens[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
seq_offset_k = seq_offset.clone()
seq_offset_t = torch.tensor([0] + [sum(num_targets[: i + 1]) for i in range(batch)], device="npu", dtype=torch.int32)
num_target = torch.tensor(num_targets, device="npu", dtype=torch.int32)

# kv_cache 分页：每个 batch 的历史 KV 按 page_size 分页存放（末页不足时以 last_page_len 标记）
# 注意：page_size 必须 ∈ {32, 64, 128, 256}（tiling 层 BLOCK_HEIGHT=256 / MIN_PAGE_SIZE=32 约束）
num_pages_per_batch = [(h + page_size - 1) // page_size for h in new_hist_lens]
page_ids = list(range(sum(num_pages_per_batch)))
page_offsets = [0] + [sum(num_pages_per_batch[: i + 1]) for i in range(batch)]
last_page_len = [h % page_size if h % page_size > 0 else page_size for h in new_hist_lens]

kv_cache = torch.randn(sum(num_pages_per_batch), 2, page_size, N, D, device="npu", dtype=torch.float16)
page_offsets = torch.tensor(page_offsets, device="npu", dtype=torch.int32)
page_ids = torch.tensor(page_ids, device="npu", dtype=torch.int32)
last_page_len = torch.tensor(last_page_len, device="npu", dtype=torch.int32)

y = ops_rec.ascendc.attention.hstu_paged(
    q, k, v, kv_cache,
    mask=None, attn_bias=None, mask_type=0,
    max_seq_len=max_seq_len, max_seq_len_k=max_seq_len, silu_scale=1.0 / max_seq_len,
    seq_offset=seq_offset, seq_offset_k=seq_offset_k, seq_offset_t=seq_offset_t,
    page_offsets=page_offsets, page_ids=page_ids, last_page_len=last_page_len,
    num_target=num_target, target_group_size=1)
print(y.shape)  # torch.Size([90, 4, 32])
```

### FP8 支持（仅 Atlas 950）

```python
# fp8_e4m3fn 输入 -> float16 输出；仅 Atlas 950（A5）支持
B, S, N, D = 2, 128, 8, 64
q = torch.randn(B, S, N, D).to(dtype=torch.float8_e4m3fn).npu()
k = torch.randn(B, S, N, D).to(dtype=torch.float8_e4m3fn).npu()
v = torch.randn(B, S, N, D).to(dtype=torch.float8_e4m3fn).npu()
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
print(y.dtype)  # torch.float16
```

## GQA (Grouped Query Attention) 支持说明

| 配置名称 | N_q | N_k | h_h_k_ratio | 说明 |
|---------|-----|-----|-------------|------|
| 标准 MHA | 8   | 8   | 1           | 每个 Q 头有独立的 K/V 头 |
| GQA-4   | 8   | 2   | 4           | 每 4 个 Q 头共享 1 个 K/V 头 |
| MQA     | 8   | 1   | 8           | 所有 Q 头共享 1 个 K/V 头 |

约束：`N_q % N_k == 0` 且 `N_k == N_v`。

## qk 与 v 的 dim 不等支持

算子支持 qk 与 v 的 dim 不等的输入，输出 dim 与 v_dim 保持一致：

```python
q = torch.randn(batch_size * seq_len, num_heads, 64, dtype=torch.float16).npu()   # head_dim_qk=64
v = torch.randn(batch_size * seq_len, num_heads, 128, dtype=torch.float16).npu()  # head_dim_v=128
y = ops_rec.ascendc.attention.hstu_jagged.equal(q, q, v, None, None, 0, 256, 1.0/256, seq_offsets_q)
print(y.shape)  # torch.Size([B*S, N, 128])
```

## 编译与验证

```bash
# 定向编译安装（atlas_950 / atlas_a2 / atlas_a3 均支持；--ops 匹配子组目录名 hstu_v1）
bash build.sh --install --socs atlas_950 --ops hstu_v1 -j8

# 运行样例（--ops 匹配 example 路径中 vendor 名 hstu_forward_v1）
bash run_examples.sh --type torch --ops hstu_forward_v1

# pytest 回归
cd ascendc/attention/hstu_v1/hstu_forward_v1/test/pytest
python3 -m pytest test_hstu_forward.py -x -v
```

## 目录结构

```text
hstu_v1/                          # HSTU 子组（转发层，登记名：hstu_v1）
└── hstu_forward_v1/              # HSTU 前向 vendor（VENDOR_NAME：hstu_forward_v1）
    ├── op_host/                  # Host 侧实现（HstuDenseForward / HstuJaggedForward / HstuPagedForward）
    │   ├── atlas950/             # A5 特化 host 文件（c310 tiling key）
    │   └── atlasA2/              # A2/A3 特化 host 文件（v220 tiling key）
    ├── op_kernel/
    │   ├── atlas950/             # A5 kernel（c310 实现）
    │   └── atlasA2/              # A2/A3 kernel（v220 实现）
    ├── torch_extension/          # PyTorch 适配层（TORCH_LIBRARY ops_rec 注册，hstu_forward.cpp）
    ├── examples/                 # aclnn (C++) 与 PyTorch (Python) 验证样例
    ├── docs/                     # 算子接口文档
    └── test/pytest/              # Pytest 自动化回归用例
```

## 迁移说明

| RecSDK 原始路径 | ops-rec 目标路径 | 调整说明 |
|-----------------|------------------|----------|
| `cust_op/ascendc_op/ai_core_op/hstu_dense_forward/v220/op_host/*` | `hstu_forward_v1/op_host/` | 移除 SUPPORT_V200（310P 专用）分支；tiling key 头下沉至 `op_host/atlasA2/` |
| `cust_op/ascendc_op/ai_core_op/hstu_dense_forward/v220/op_kernel/*` | `hstu_forward_v1/op_kernel/atlasA2/` | 移除 SUPPORT_V200（310P 专用）分支 |
| `cust_op/ascendc_op/ai_core_op/hstu_dense_forward/c310/op_kernel/*` | `hstu_forward_v1/op_kernel/atlas950/` | A5 特化 kernel，SUPPORT_950 由根 CMake 注入 |
| `cust_op/ascendc_op/ai_core_op/hstu_dense_forward/c310/op_kernel/hstu_kernel_tiling_key.h` | `hstu_forward_v1/op_host/atlas950/` | host 侧 tiling 生成依赖 c310 tiling key 布局 |
| `cust_op/framework/torch_plugin/torch_library/hstu/*` | `torch_extension/hstu_forward.cpp` | 合并三文件，命名空间 mxrec → ops_rec，补充 Meta 注册 |
