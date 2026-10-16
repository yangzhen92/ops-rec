# torch.ops.ops_rec.hstu_forward_v2

## 功能说明

HSTU (Hierarchical Sparse Transformer Unit) Forward V2 算子的 PyTorch 接口，计算 HSTU 注意力的前向输出。v2 版本统一使用 jagged 格式输入，通过 seq_offset 管理变长序列。

计算流程：

$$
S = \alpha \cdot (Q K^\top + \text{rab})
$$

$$
O = \text{SiLU}(S) \cdot \text{silu\_scale} \cdot V
$$

其中 `rab` 为可选的 Relative Attention Bias，`silu_scale` 默认取 `1.0 / max_seq_len`。mask 依据 `mask_type`、`window_size_left/right`、`sparse_info` 在 SiLU 之后作用到注意力分数上。

## 接口定义

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

## 参数说明

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **mask** (*Tensor, optional*) – 自定义 mask 张量，`mask_type=3` 时必须提供。乘性 0/1 掩码，形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, 1|heads, maxSeqLenQ, maxSeqLenK]`，head 为 1 时广播。

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

- **metadata** (*Tensor, optional*) – 预生成的分核 metadata（int32，HEAD+FA+FD 布局，如 `create_attn_metadata` 输出）。默认值: `None`（kernel 内现算分核）

- **sparse_info** (*Tensor[], optional*) – 任意 mask 的压缩表示，由 `get_fwd_attn_plan` 生成。必须恰好包含 9 个 int32 张量。默认值: `None`

- **window_size_left** (*int*) – 注意力窗口左侧宽度，默认 `-1`。`-1` 表示向左无限延伸。与 `window_size_right` 共同决定 mask 类型

- **window_size_right** (*int*) – 注意力窗口右侧宽度，默认 `-1`。`-1` 表示向右无限延伸，`0` 表示因果掩码。与 `window_size_left` 共同决定 mask 类型

- **plan_af** (*Tensor, optional*) – 预留参数。默认值: `None`

- **deterministic** (*bool, optional*) – 是否确定性累加。默认值: `True`。设为 `False` 时采用 Stream-K 非确定性归约（内部自动创建 float32 累加缓冲），性能更优但结果可能存在微小误差

## 支持的数据类型

`torch.float16, torch.bfloat16`

## Shape

- **输入**:
    - `q`: `(totalSeqLenQ, heads, dimQK)`
    - `k`: `(totalSeqLenK, heads, dimQK)`
    - `v`: `(totalSeqLenK, heads, dimGV)`
    - `mask` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`
    - `rab` (optional): `(batchSize, 1|heads, maxSeqLenQ, maxSeqLenK)`
    - `seq_offset`: `(batchSize + 1)`
    - `seq_offset_k` (optional): `(batchSize + 1)`
    - `num_context`/`num_target` (optional): `(batchSize)`

- **输出**: 单个 Tensor
    - `attnOutput`: `(totalSeqLenQ, heads, dimGV)`，dtype 与输入一致

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型，`q/k/v` 必须为 3D 且 dtype 一致
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数且 ≤ 256
- `seq_offset` 必须为 1D `int32` 张量，长度 `batchSize + 1`；`seq_offset_k`（如提供）长度必须与 `seq_offset` 一致
- `batchSize` 必须大于 `0`，`max_seq_len` 与 `max_seq_len_k` 必须大于 `0`
- `k` 的 `heads/dimQK` 必须与 `q` 一致；`v` 的行数必须与 `k` 一致、`heads` 必须与 `q` 一致
- `mask_type` 取值范围 `[0, 3]`，不支持 `1`（triu）；`mask_type=3` 时必须提供 `mask` 张量
- `window_size_left` 和 `window_size_right` 必须 ≥ `-1` 且 `< max_seq_len_k`
- 非局部窗口（`window_size_left = -1` 且 `window_size_right ≤ 0`）时，`mask_type` 只能为 `0`（causal）或 `2`（none）
- 当提供 `num_context` 或 `num_target` 时，必须使用 causal 或 local mask（与 backward 一致）
- `alpha` 必须大于 `0`
- `sparse_info` 提供时必须恰好包含 9 个张量

## mask 行为说明

通过 `mask_type` 与 `window_size_left`、`window_size_right` 共同控制注意力 mask 类型：

| mask_type | 含义 | 说明 |
|-----------|------|------|
| 0 | causal mask（因果掩码） | 下三角掩码，每个 token 只能关注自身及之前的 token |
| 1 | triu | 不支持 |
| 2 | no mask（无掩码） | 不做任何 mask，每个 token 可关注所有 token |
| 3 | custom mask | 乘性自定义 mask，必须提供形状 `[batchSize, heads, maxSeqLenQ, maxSeqLenK]` 的 mask 张量 |

通过 `window_size_left` 和 `window_size_right` 还可实现局部窗口（local window）注意力：`window_size_left ≥ 0` 或 `window_size_right > 0` 时，query 仅关注左侧 `window_size_left` 和右侧 `window_size_right` 范围。

当使用 causal mask 时，可通过 `num_context` 和 `num_target` 进一步细分 mask 区域：

- **context 区域**（`num_context` 指定长度）：双向注意力，无 causal 限制
- **target 区域**（`num_target` 指定长度）：分组注意力，group_size 由 `target_group_size` 控制
- **其余区域**：标准 causal mask（下三角）

任意 mask 亦可不传入稠密 mask 张量，而是通过 `get_fwd_attn_plan` 生成的 `sparse_info`（9 个打包张量）压缩表示传入。

注意：`mask_type` 与 `window_size` 的合法组合见「约束条件」；`num_context`/`num_target` 仅用于 causal 场景，不与自定义 mask 张量组合。

## 图模式（inductor）支持

`hstu_forward_v2` 注册了 Meta 实现（`TORCH_LIBRARY_IMPL(ops_rec, Meta, m)`），支持 `torch.compile` 构图。

> 注：Python ≥ 3.12 下 `torch.compile` 存在兼容问题，测试用例通过 `torch_compiled` 封装降级为 eager 执行；图模式仅在 Python < 3.12 下生效。具体降级逻辑见 `test/pytest/` 中的 `torch_compiled` 函数。

```python
compiled = torch.compile(
    torch.ops.ops_rec.hstu_forward_v2,
    dynamic=True,
    fullgraph=True,
)
output = compiled(
    q, k, v, mask, rab,
    mask_type, max_seq_len, max_seq_len_k, silu_scale,
    seq_offset, seq_offset_k, num_context, num_target,
    target_group_size, alpha,
    None, None, -1, -1, None, True,
)
```

注意：`num_context` 和 `num_target` 的 schema 类型为 `Tensor?`，传非 `None` 值时需构造为 `[batchSize]` 的 `int32` Tensor 传入，不可直接传 `int`。

图模式运行依赖 `PrivateUse1 + Meta` 干净注册路径；Meta 实现的输出 shape 与真实 kernel 对齐（`attnOutput` 形状为 `q` 的前两维 + `v` 的最后一维）。回归用例见 `test/pytest/test_forward_precision.py`（`@pytest.mark.graph` 相关用例）。

## 调用示例

```python
import torch
import ops_rec

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False

batch_size, heads, dim_qk, dim_gv = 2, 4, 64, 64
max_seqlen_q, max_seqlen_k = 128, 128
total_seq_len_q, total_seq_len_k = 200, 200

q = torch.rand(total_seq_len_q, heads, dim_qk, dtype=torch.float16).uniform_(-1, 1).to("npu")
k = torch.rand(total_seq_len_k, heads, dim_qk, dtype=torch.float16).uniform_(-1, 1).to("npu")
v = torch.rand(total_seq_len_k, heads, dim_gv, dtype=torch.float16).uniform_(-1, 1).to("npu")
rab = torch.rand(batch_size, heads, max_seqlen_q, max_seqlen_k, dtype=torch.float16).uniform_(-1, 1).to("npu")

seq_lens_q = torch.tensor([100, 100], dtype=torch.int32)
seq_lens_k = torch.tensor([100, 100], dtype=torch.int32)
seq_offset_q = torch.concat((torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_q, 0))).to("npu")
seq_offset_k = torch.concat((torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_k, 0))).to("npu")

# causal mask: mask_type=0；无掩码: mask_type=2
output = torch.ops.ops_rec.hstu_forward_v2(
    q, k, v, None, rab,
    0,                              # mask_type: causal
    max_seqlen_q, max_seqlen_k,
    1.0 / max_seqlen_q,             # silu_scale
    seq_offset_q, seq_offset_k,
    None, None,                     # num_context / num_target
    1, 0.5,                         # target_group_size / alpha
)
torch.npu.synchronize()
print(output.shape)  # torch.Size([200, 4, 64])
```

更多用例参考 [test_forward_precision.py](../test/pytest/test_forward_precision.py)，调用封装参考 [ascend_fuse_backend.py](../test/pytest/backend/ascend_fuse_backend.py)。
