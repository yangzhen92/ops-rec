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

**FP8 支持**：本算子仅在 A5 上支持 data_type 为 fp8_e4m3fn 的 qkv 输入，算子输出为 float16。

## 算子实现原理

### 计算公式

$$
HSTU(q, k, v, mask, bias, siluScale) = (Silu(qk^{T} + bias) \times siluScale \times mask)v
$$

### 数据格式

输入参数 q, k, v 数据格式为 normal、jagged 或 paged：

* **normal（dense）格式**：shape 为 `[B, S, N, D]` 的 4 维数据
* **jagged 格式**：shape 为 `[s_b, N, D]` 的 3 维数据，配合 `seq_offset_q`/`seq_offset_k` 使用
* **paged 格式**：3 维数据配合 `kv_cache` 分页存储使用

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

```bash
# PyTorch（安装 whl 后）
import ops_rec
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None,
                                         mask_type=0, max_seq_len=256, silu_scale=1.0/256)

# jagged 格式
y = ops_rec.ascendc.attention.hstu_jagged.equal(
    q, k, v, None, None, mask_type=0, max_seq_len=256, silu_scale=1.0/256,
    seq_offset=seq_offsets_q, num_context=None, num_target=None)
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
