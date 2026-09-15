# hstu_backward_v2

## 产品支持情况

| 硬件型号 | 是否支持 |
| -------- |------|
| Ascend 950PR&950DT 系列产品 | 是    |

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

HSTU (Hierarchical Sparse Transformer Unit) Backward V2 算子的反向传播实现，用于计算 Query、Key、Value 以及 RAB 注意力分数的梯度。

该算子实现 Transformer 注意力机制的反向传播，通过 CATLASS 模板库在昇腾 NPU 上高效执行矩阵乘法运算。

## 接口说明

| 接口类型 | 接口名称 | 说明 |
| -------- | -------- | ---- |
| aclnn | `aclnnHstuBackwardV2` | 反向算子 |
| torch | `ops_rec.ascendc.attention.hstu_backward_v2` | PyTorch 接口 |

## 参数说明

- **grad** (*Tensor*) – 输出梯度张量。形状: `[totalSeqLenQ, heads, dimGV]`，数据类型: `float16, bfloat16`

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **max_seqlen_q** (*int*) – Query 的最大序列长度

- **max_seqlen_k** (*int*) – Key/Value 的最大序列长度

- **seq_offset_q** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **seq_offset_k** (*Tensor*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`，数据类型: `float16, bfloat16`。默认值: `None`

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
    - `rab` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`

- **输出**: 4 个 Tensor 的元组
    - `qGrad`: `(totalSeqLenQ, heads, dimQK)`，dtype: `float16, bfloat16`
    - `kGrad`: `(totalSeqLenK, heads, dimQK)`，dtype: `float16, bfloat16`
    - `vGrad`: `(totalSeqLenK, heads, dimGV)`，dtype: `float16, bfloat16`
    - `rabGrad`: `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`，dtype: `float16, bfloat16`（仅当 `rab` 不为 None 时返回）

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数
- 当 `num_context` 或 `num_target` 不为 None 时，`target_group_size` 必须为 `1` 或 `3`
- `headQ` 必须等于 `headK`（当前仅支持 MHA，不支持 GQA）
- `window_size_left` 和 `window_size_right` 必须 ≥ -1
- 当前仅支持 `(-1, 0)`（causal mask）和 `(-1, -1)`（no mask）两种 `(window_size_left, window_size_right)` 组合
- 当提供 `num_context` 或 `num_target` 时，必须使用 causal mask（`window_size_left=-1, window_size_right=0`）, 同时需满足num_context + num_target < max_seqlen_q

## mask 行为说明

通过 `window_size_left` 和 `window_size_right` 控制注意力 mask 类型：

| window_size_left | window_size_right | mask 类型 | 说明 |
|-----------------|-------------------|----------|------|
| -1 | 0 | causal mask（因果掩码） | 下三角掩码，每个 token 只能关注自身及之前的 token |
| -1 | -1 | no mask（无掩码） | 不做任何 mask，每个 token 可关注所有 token |

当使用 causal mask 时，可通过 `num_context` 和 `num_target` 进一步细分 mask 区域：

- **context 区域**（`num_context` 指定长度）：双向注意力，无 causal 限制
- **target 区域**（`num_target` 指定长度）：分组注意力，group_size 由 `target_group_size` 控制
- **其余区域**：标准 causal mask（下三角）

注意：当前仅支持 causal mask 和 no mask 两种模式，不支持自定义 mask 张量输入。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_hstu_backward_v2.cpp)
- [torch 调用示例](./examples/torch/test_torch_hstu_backward_v2.py)
- [aclnn 接口文档](./docs/aclnn_hstu_backward_v2.md)
- [torch 接口文档](./docs/torch_hstu_backward_v2.md)
