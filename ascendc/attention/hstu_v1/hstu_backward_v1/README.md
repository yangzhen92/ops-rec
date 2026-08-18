# hstu_backward_v1

## 产品支持情况

| 硬件型号 | 是否支持 |
| -------- | -------- |
| Atlas A2 训练系列产品 | 是 |
| Atlas A3 训练系列产品 | 是 |
| Atlas 950 训练系列产品 | 是 |

## 目录层级

```text
hstu_backward_v1/
├── CMakeLists.txt           # 根目录 CMake 配置
├── op_host/                 # Host 侧实现
├── op_kernel/               # Kernel 侧实现
├── examples/                # 调用示例
│   ├── aclnn/               # aclnn 调用示例
│   └── torch/               # torch 调用示例
├── test/                    # 测试用例
│   └── pytest/              # pytest 测试
├── docs/                    # 接口文档
└── README.md
```

## 功能说明

算子的主要功能是实现 HSTU 融合算子的反向传播 (`hstu_backward`)。

支持两种布局：

- **Normal 布局**：输入 shape [B, S, N, D]，适用于固定长度序列
- **Jagged 布局**：输入 shape [T, N, D]，适用于变长序列，其中 T 为所有 batch 的序列长度之和

## 算子实现原理

### Normal Layout

1. 计算 QK 和 GV 矩阵乘法
2. 转换为 float 类型进行计算（提高精度）
3. 添加 attention bias（如提供）
4. 应用 silu_scale 并计算 attention score
5. 计算 V 的梯度
6. 计算 attention bias 的梯度（SiLU 的导数）
7. 计算 K 和 Q 的梯度

### Jagged Layout

Jagged Layout 的实现原理与 Normal Layout 类似，主要区别在于：

1. **数据格式转换**：需要将 jagged 格式转换为 dense 格式进行计算
2. **Alpha 参数**：在计算 `qkb` 和 `attn_bias_grad` 时需要乘以 `alpha` 参数
3. **结果转换**：计算完成后需要将结果从 dense 格式转换回 jagged 格式

## 接口说明

| 接口类型 | 接口名称 | 说明 |
| -------- | -------- | ---- |
| aclnn | `aclnnHstuDenseBackward` | Normal 布局反向算子 |
| aclnn | `aclnnHstuJaggedBackward` | Jagged 布局反向算子 |
| torch | `ops_rec.ascendc.attention.hstu_dense_backward` | Normal 布局 PyTorch 接口 |
| torch | `ops_rec.ascendc.attention.hstu_jagged_backward` | Jagged 布局 PyTorch 接口 |

## 输入输出参数

### 输入参数

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ---- | --------- | -------- | -------- | -------- | ---- | ---- |
| grad | 输入 | Tensor (REQUIRED) | float32/float16/bf16 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[1, 16]<br>D/D_v∈[16, 512]且是16的倍数 | 前向输出 out 的反向梯度，Jagged 模式下 T 为所有 batch 的序列长度 seq 的和 |
| q | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | B、S、N、D 同 grad<br>D_qk∈[1, 512] | Q 张量 |
| k | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 同 q | K 张量，Shape 和类型与 Q 一致 |
| v | 输入 | Tensor (REQUIRED) | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | 同 grad | V 张量，Shape 和类型与 grad 一致 |
| mask | 输入 | Tensor (OPTIONAL) | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 同 grad | mask 张量，当 mask_type=3 时必须提供 |
| attn_bias | 输入 | Tensor (OPTIONAL) | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 同 grad | attn_bias 张量 |
| seq_offset_q | 输入 | Tensor (OPTIONAL) | int32/int64 | [BatchSize+1] | BatchSize∈[1, 2048] | Jagged 模式下必须提供，表示每个 batch 的序列长度偏移 |
| seq_offset_k | 输入 | Tensor (OPTIONAL) | int32/int64 | [BatchSize+1] | BatchSize∈[1, 2048] | Jagged 模式下必须提供 |
| num_context | 输入 | Tensor (OPTIONAL) | int32/int64 | [B] | B∈[1, 2048]，值范围[1, 128] | Context 掩码长度 |
| num_target | 输入 | Tensor (OPTIONAL) | int32/int64 | [B] | B∈[1, 2048]，值范围[1, 512] | Target 掩码长度 |

### 属性参数

| 名称 | 参数类型 | 数据类型 | 默认值 | 范围/取值 | 说明 |
| ---- | -------- | -------- | ------ | --------- | ---- |
| mask_type | Attr | int | - | [0, 2, 3] | mask 类型：0 表示使用内置下三角 mask（TRIL），2 表示不使用 mask，3 表示使用自定义 mask（CUSTOM）<br>注意：1（TRIU）当前不支持 |
| max_seq_len_q | Attr | int | - | [1, 20480] | 模型最大序列长度 |
| max_seq_len_k | Attr | int | - | [1, 20480] | 模型最大序列长度 |
| silu_scale | Attr | float | 0.0 | 任意值 | SiLU 激活函数前的缩放因子，如果为 0.0 则自动计算为 1.0/max_seq_len |
| target_group_size | Attr (OPTIONAL) | int | None | {1, 3} | Target 分组大小 |
| alpha | Attr (OPTIONAL) | float | 1.0 | 任意值 | 缩放因子 |

### 输出参数

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ---- | --------- | -------- | -------- | -------- | ---- | ---- |
| q_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 与 q 保持一致 | Q 的梯度 |
| k_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_qk] | 与 k 保持一致 | K 的梯度 |
| v_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | Normal: [B, S, N, D]<br>Jagged: [T, N, D_v] | 与 v 保持一致 | V 的梯度 |
| attn_bias_grad | 输出 | Tensor | float32/float16/bf16<br>与 grad 类型一致 | [B, N, S, S] | 与 attn_bias 保持一致 | attn_bias 的反向梯度，如果未提供 attn_bias 则返回空张量 |

## 约束说明

### 数据类型约束

- grad、q、k、v、mask、attn_bias 的数据类型必须一致，支持 FLOAT32、FLOAT16、BF16
- seq_offset_q、seq_offset_k、num_context、num_target 的数据类型为 INT32 或 INT64

### 维度约束

- Normal 布局：输入必须为 4D [B, S, N, D]
- Jagged 布局：输入必须为 3D [T, N, D]，其中 T 为所有 batch 的序列长度之和
- mask 和 attn_bias（若提供）必须为 4D [B, N, S, S]
- head_dim 必须为 16 的倍数
- N_q 必须能被 N_kv 整除（GQA 场景）

### 数值范围约束

- mask_type 支持：0 (TRIL), 2 (NONE), 3 (CUSTOM)
- B ∈ [1, 2048]
- S ∈ [1, 20480]
- N ∈ [1, 16]
- D_qk ∈ [16, 512]
- D_v ∈ [16, 512]
- num_context 中每个值 ∈ [1, 128]，需满足 num_context + num_target < max_seq_len
- num_target 中每个值 ∈ [1, 512]
- target_group_size ∈ {1, 3}

### 反向传播特殊约束

- Normal 模式下，seqLen 必须等于 max_seq_len
- Jagged 模式下，seqLen 必须与 mask 和 attn_bias 的 shape 中 S 相等
- 反向传播时，mask 和 attn_bias 的 S 维必须与 max_seq_len 一致

### 复合 Mask（Causal + Context + Target）约束

复合 Mask 是 HSTU 模型专用的 mask 模式，将序列划分为三个区域：

```text
序列结构: [context] [causal] [target]
```

- **Context 区域**（前 num_context 个 token）：各 token 可关注所有 context token（全量注意力）
- **Causal 区域**（中间 token）：遵循因果注意力（只关注 context 和自身之前的 causal token）
- **Target 区域**（后 num_target 个 token）：只关注 context 和自身（不跨 target 相互关注）

启用复合 Mask 时，必须同时提供以下三个参数：

- `num_context`：每个 batch 的 context 长度，shape [B]
- `num_target`：每个 batch 的 target 长度，shape [B]
- `target_group_size`：target 分组大小，取值 {1, 3}

### 设备约束

- 所有输入 Tensor 必须在同一 NPU 设备上
- 该算子仅在 NPU 设备上运行

## 算子执行样例

### 安装算子模块

```bash
cd ops-rec
bash build.sh --install --ops=hstu_backward_v1
```

### 导入依赖

```python
import torch
import ops_rec

torch.npu.set_device(0)
```

以下示例使用 PyTorch 接口演示 `hstu_jagged_backward` 算子的常见用法。每个示例在上一示例基础上做增量改动。

### 1. 基础场景（MHA，Jagged 布局）

MHA 模式，无 mask，无 attention bias。使用 Jagged 布局处理变长序列。

> **序列结构说明**：Jagged 布局将多个变长序列打包为一个 3D 张量 [T, N, D]，其中 T 为所有序列长度之和。通过 `seq_offset_q` 和 `seq_offset_k` 记录每个序列的起始偏移。

```python
# ===================== 示例 1: 基础场景（MHA） =====================
# MHA 模式：nheads == nheads_kv
# mask_type=2：不使用 mask
# 无 attn_bias

# 变长序列参数
batch_size = 2                          # batch 数
seq_lens = [6, 10]                      # 两个 batch 的序列长度
total_len = sum(seq_lens)               # 总 token 数 = 16
max_seq_len = max(seq_lens)             # 最大序列长度 = 10
head_num_q = 4                          # Q 头数
head_num_k = 4                          # K/V 头数
head_dim_qk = 32                        # Q/K 的 head_dim
head_dim_v = 32                         # V 的 head_dim
data_type = torch.float32
mask_type = 2                           # 不使用 mask
mask = None
attn_bias = None
silu_scale = 0.0
num_context = None
num_target = None
target_group_size = 0
alpha = 1.0

# 构造 seq_offset：每个 batch 在 jagged 张量中的起始偏移
# 格式：[0, seq0_len, seq0_len+seq1_len, ...]
seq_offset = torch.tensor([0, 6, 16], dtype=torch.int32).npu()

# 构造 jagged 张量：[total_len, N, D]
grad = torch.empty(total_len, head_num_q, head_dim_v, dtype=data_type).uniform_(-1, 1).npu()
q    = torch.empty(total_len, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1).npu()
k    = torch.empty(total_len, head_num_k, head_dim_qk, dtype=data_type).uniform_(-1, 1).npu()
v    = torch.empty(total_len, head_num_k, head_dim_v, dtype=data_type).uniform_(-1, 1).npu()

# 调用 hstu_jagged_backward
q_grad, k_grad, v_grad, _ = ops_rec.ascendc.attention.hstu_jagged_backward(
    grad, q, k, v,
    mask, attn_bias,
    mask_type=mask_type,
    max_seq_len_q=max_seq_len,
    max_seq_len_k=max_seq_len,
    silu_scale=silu_scale,
    seq_offset_q=seq_offset,
    seq_offset_k=seq_offset,
    num_context=num_context,
    num_target=num_target,
    target_group_size=target_group_size,
    alpha=alpha,
)
print(f"q_grad: {q_grad.shape}, k_grad: {k_grad.shape}, v_grad: {v_grad.shape}")
```

### 2. GQA（分组查询注意力）

**相比示例 1 的改动：**

- Query 头数（`head_num_q`）大于 Key/Value 头数（`head_num_k`），GQA ratio = 2
- `grad` 和 `q` 使用 `head_num_q` 头，`k` 和 `v` 使用 `head_num_k` 头

```python
# ===================== 示例 2: GQA =====================
# 相比示例 1 的改动：GQA 模式（head_num_q > head_num_k）
# 其余参数和调用方式与示例 1 相同

head_num_q = 8                          # 8 个 query 头
head_num_k = 4                          # 4 个 key/value 头（GQA ratio=2）

# grad 和 q 使用 head_num_q 头，k 和 v 使用 head_num_k 头
grad = torch.empty(T, head_num_q, head_dim_v, dtype=data_type).uniform_(-1, 1).npu()
q    = torch.empty(T, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1).npu()
k    = torch.empty(T, head_num_k, head_dim_qk, dtype=data_type).uniform_(-1, 1).npu()
v    = torch.empty(T, head_num_k, head_dim_v, dtype=data_type).uniform_(-1, 1).npu()
```

### 3. 复合 Mask（Causal + Context + Target）

**相比示例 2 的改动：**

- 新增 `num_context`、`num_target`、`target_group_size` 参数
- `mask_type` 改为 0（TRIL），算子使用内置下三角 mask 并结合 `num_context`/`num_target`/`target_group_size` 在内部构建 HSTU 复合 mask

> **复合 Mask 模式说明**：序列被划分为三个区域：
> 
> - **Context 区域**（前 num_context 个 token）：全量关注所有 context token
> - **Causal 区域**（中间 token）：因果关注（context + 自身之前的 causal token）
> - **Target 区域**（后 num_target 个 token）：只关注 context 和自身（不跨 target 关注）

```python
# ===================== 示例 3: 复合 Mask =====================
# 相比示例 2 的改动：
#   - 新增 num_context / num_target / target_group_size
#   - mask_type 改为 0（TRIL），算子内部结合复合 mask 参数构造 HSTU 专用 mask
# 其余参数和调用方式与示例 1 相同

num_context = torch.tensor([3, 4], dtype=torch.int32).npu()   # batch 0=3, batch 1=4
num_target  = torch.tensor([1, 2], dtype=torch.int32).npu()   # batch 0=1, batch 1=2
target_group_size = 1
mask_type = 0                           # 使用内置下三角 mask，算子结合 num_context 等构造复合 mask
```

## 调用示例

### aclnn 接口

- [Normal 调用示例](examples/aclnn/test_aclnn_hstu_backward_v1_dense.cpp)
- [Jagged 调用示例](examples/aclnn/test_aclnn_hstu_backward_v1_jagged.cpp)
- [Normal 调用接口文档](./docs/aclnn_hstu_dense_backward.md)
- [Jagged 调用接口文档](./docs/aclnn_hstu_jagged_backward.md)

### torch 接口

- [Normal 调用示例](examples/torch/test_torch_hstu_backward_v1_dense.py)
- [Jagged 调用示例](examples/torch/test_torch_hstu_backward_v1_jagged.py)
- [Normal 调用接口文档](./docs/torch_hstu_dense_backward.md)
- [Jagged 调用接口文档](./docs/torch_hstu_jagged_backward.md)

### pytest 测试

- [Normal 测试](./test/pytest/test_hstu_dense_backward.py)
- [Jagged 测试](./test/pytest/test_hstu_jagged_backward.py)
