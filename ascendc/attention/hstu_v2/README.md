# hstu_v2

## 产品支持情况

| 硬件型号 | 是否支持 |
| -------- |------|
| Ascend 950PR&950DT系列产品 | 是    |

## 目录层级

```test
hstu_v2/
├── CMakeLists.txt           # 根目录 CMake 配置
├── op_host/                 # Host 侧实现
├── op_kernel/               # Kernel 侧实现
├── torch_extension/         # PyTorch 扩展
├── examples/                # 调用示例
│   ├── aclnn/               # aclnn 调用示例
│   └── torch/               # torch 调用示例
├── test/                    # 测试用例
│   └── pytest/              # pytest 测试
├── docs/                    # 接口文档
└── README.md
```

## 功能说明

HSTU (Hierarchical Sparse Transformer Unit) V2 算子套件，包含 Forward 与 Backward 两个算子，统一使用 jagged 格式输入，通过 seq_offset 管理变长序列。

- **hstu_forward_v2**：前向注意力计算，输出 HSTU 注意力结果（SiLU 缩放注意力得分与 V 的矩阵乘）。
- **hstu_backward_v2**：反向传播实现，用于计算 Query、Key、Value 以及 RAB 注意力分数的梯度。

两个算子均通过 CATLASS 模板库在昇腾 NPU 上高效执行矩阵乘法运算。

## 接口说明

| 接口类型 | 接口名称 | 说明 |
| -------- | -------- | ---- |
| aclnn | `aclnnHstuForwardV2` | 前向算子 |
| torch | `torch.ops.ops_rec.hstu_forward_v2` | PyTorch 接口 |
| aclnn | `aclnnHstuBackwardV2` | 反向算子 |
| torch | `torch.ops.ops_rec.hstu_backward_v2` | PyTorch 接口 |

## 参数说明

### hstu_forward_v2

```python
torch.ops.ops_rec.hstu_forward_v2(
    q, k, v, mask, rab,
    mask_type, max_seq_len, max_seq_len_k, silu_scale,
    seq_offset, seq_offset_k, num_context, num_target,
    target_group_size, alpha,
    metadata=None, sparse_info=None,
    window_size_left=-1, window_size_right=-1,
    plan_af=None, deterministic=True
) -> Tensor
```

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **mask** (*Tensor, optional*) – 自定义 mask 张量，`mask_type=3` 时必须提供。乘性 0/1 掩码，形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, rabHeads, maxSeqLenQ, maxSeqLenK]`，其中 `rabHeads` 为 `1`（跨 attention head 广播）或 `heads`，数据类型: `float16, bfloat16`

- **mask_type** (*int*) – mask 类型。`0`: 因果掩码（下三角）；`1`: 上三角，不支持；`2`: 无掩码；`3`: 自定义 mask（需提供 `mask` 张量）

- **max_seq_len** (*int*) – Query 的最大序列长度

- **max_seq_len_k** (*int, optional*) – Key/Value 的最大序列长度。默认值: `None`（取 `max_seq_len`）

- **silu_scale** (*float, optional*) – SiLU 之后的缩放因子。默认值: `None`（实际使用 `1.0 / maxSeqLenQ`）

- **seq_offset** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **seq_offset_k** (*Tensor, optional*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`。默认值: `None`（复用 `seq_offset`）

- **num_context** (*Tensor, optional*) – 每个 batch 的 Context 长度。形状: `[batchSize]`。默认值: `None`

- **num_target** (*Tensor, optional*) – 每个 batch 的 Target 长度。形状: `[batchSize]`。默认值: `None`

- **target_group_size** (*int, optional*) – RAG 目标组大小。默认值: `0`

- **alpha** (*float, optional*) – Alpha 系数，作用于 `QK^T + rab`。必须大于 `0`。默认值: `1.0`

- **metadata** (*Tensor, optional*) – 预生成的分核 metadata（int32）。默认值: `None`（kernel 内现算分核）

- **sparse_info** (*Tensor[], optional*) – 任意 mask 的压缩表示，由 `get_fwd_attn_plan` 生成，必须恰好包含 9 个 int32 张量。默认值: `None`

- **window_size_left** (*int*) – 注意力窗口左侧宽度，默认 `-1`。`-1` 表示向左无限延伸。与 `window_size_right` 共同决定 mask 类型

- **window_size_right** (*int*) – 注意力窗口右侧宽度，默认 `-1`。`-1` 表示向右无限延伸，`0` 表示因果掩码。与 `window_size_left` 共同决定 mask 类型

- **plan_af** (*Tensor, optional*) – 预留参数。默认值: `None`

- **deterministic** (*bool, optional*) – 是否确定性累加。默认值: `True`。设为 `False` 时采用 Stream-K 非确定性归约，性能更优但结果可能存在微小误差

输出: 单个 Tensor，`attnOutput`: `(totalSeqLenQ, heads, dimGV)`，dtype 与输入一致。

### hstu_backward_v2

- **grad** (*Tensor*) – 输出梯度张量。形状: `[totalSeqLenQ, heads, dimGV]`，数据类型: `float16, bfloat16`

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **max_seqlen_q** (*int*) – Query 的最大序列长度

- **max_seqlen_k** (*int*) – Key/Value 的最大序列长度

- **seq_offset_q** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **seq_offset_k** (*Tensor*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, rabHeads, maxSeqLenQ, maxSeqLenK]`，其中 `rabHeads` 为 `1`（跨 attention head 广播）或 `heads`，数据类型: `float16, bfloat16`。默认值: `None`

- **num_context** (*Tensor, optional*) – 每个 batch 的 Context 长度。形状: `[batchSize]`。默认值: `None`

- **num_target** (*Tensor, optional*) – 每个 batch 的 Target 长度。形状: `[batchSize]`。默认值: `None`

- **scale** (*float, optional*) – 缩放因子。默认值: `0.0`（实际使用 `1.0 / maxSeqLenQ`）

- **target_group_size** (*int, optional*) – RAG 目标组大小。必须为 `1` 或 `3`。默认值: `0`

- **alpha** (*float, optional*) – Alpha 系数，用于 RAG 计算。默认值: `1.0`

- **window_size_left** (*int*) – 注意力窗口左侧宽度，默认 `-1`。`-1` 表示向左无限延伸。与 `window_size_right` 共同决定 mask 类型

- **window_size_right** (*int*) – 注意力窗口右侧宽度，默认 `-1`。`-1` 表示向右无限延伸，`0` 表示因果掩码。与 `window_size_left` 共同决定 mask 类型

## 支持的数据类型

`torch.float16, torch.bfloat16`

## Shape

- **输入**:
    - `grad`: `(totalSeqLenQ, heads, dimGV)`
    - `q`: `(totalSeqLenQ, heads, dimQK)`
    - `k`: `(totalSeqLenK, heads, dimQK)`
    - `v`: `(totalSeqLenK, heads, dimGV)`
    - `seq_offset_q`: `(batchSize + 1)`
    - `seq_offset_k`: `(batchSize + 1)`
    - `rab` (optional): `(batchSize, 1|heads, maxSeqLenQ, maxSeqLenK)`

- **输出**: 4 个 Tensor 的元组
    - `qGrad`: `(totalSeqLenQ, heads, dimQK)`，dtype: `float16, bfloat16`
    - `kGrad`: `(totalSeqLenK, heads, dimQK)`，dtype: `float16, bfloat16`
    - `vGrad`: `(totalSeqLenK, heads, dimGV)`，dtype: `float16, bfloat16`
    - `rabGrad`: 与 `rab` shape 相同，dtype: `float16, bfloat16`（仅当 `rab` 不为 None 时返回）

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数
- 当 `num_context` 或 `num_target` 不为 None 时，`target_group_size` 必须为 `1` 或 `3`
- `headQ` 必须等于 `headK`（当前仅支持 MHA，不支持 GQA）
- `window_size_left` 和 `window_size_right` 必须 ≥ -1
- 支持 `(-1, 0)` causal、`(-1, -1)` no-mask，以及任一边为 `-1` 或非负整数的 local window
- 当提供 `num_context` 或 `num_target` 时，必须使用 causal mask（`window_size_left=-1, window_size_right=0`）, 同时需满足num_context + num_target < max_seqlen_q

## mask 行为说明

通过 `window_size_left` 和 `window_size_right` 控制注意力 mask 类型：

| window_size_left | window_size_right | mask 类型 | 说明 |
|-----------------|-------------------|----------|------|
| -1 | 0 | causal mask（因果掩码） | 下三角掩码，每个 token 只能关注自身及之前的 token |
| -1 | -1 | no mask（无掩码） | 不做任何 mask，每个 token 可关注所有 token |
| >=0 | >=-1 | local window | query 仅关注左侧 `window_size_left` 和右侧 `window_size_right` 范围 |

当使用 causal mask 时，可通过 `num_context` 和 `num_target` 进一步细分 mask 区域：

- **context 区域**（`num_context` 指定长度）：双向注意力，无 causal 限制
- **target 区域**（`num_target` 指定长度）：分组注意力，group_size 由 `target_group_size` 控制
- **其余区域**：标准 causal mask（下三角）

注意：`num_context`/`num_target` 仅用于 `(-1,0)` causal；任意 local window 不与它们组合。

## 调用示例

- [aclnn 调用示例（backward）](./examples/aclnn/test_aclnn_hstu_backward_v2.cpp)
- [torch 调用示例（backward）](./examples/torch/test_torch_hstu_backward_v2.py)
- [torch 精度用例（forward）](./test/pytest/test_forward_precision.py)
- [aclnn 接口文档（forward）](./docs/aclnn_hstu_forward_v2.md)
- [torch 接口文档（forward）](./docs/torch_hstu_forward_v2.md)
- [aclnn 接口文档（backward）](./docs/aclnn_hstu_backward_v2.md)
- [torch 接口文档（backward）](./docs/torch_hstu_backward_v2.md)
