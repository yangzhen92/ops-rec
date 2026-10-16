# aclnnHstuForwardV2

## 功能说明

HSTU (Hierarchical Sparse Transformer Unit) Forward V2 算子的 aclnn 接口，计算 HSTU 注意力的前向输出。v2 版本统一使用 jagged 格式输入，通过 seq_offset 管理变长序列。

计算流程：

$$
S = \alpha \cdot (Q K^\top + \text{rab})
$$

$$
O = \text{SiLU}(S) \cdot \text{siluScale} \cdot V
$$

其中 `rab` 为可选的 Relative Attention Bias，`siluScale` 取 `0` 时实际使用 `1.0 / maxSeqLenQ`。mask 依据 `maskType`、`windowSizeLeft/Right`、`sparseInfo` 在 SiLU 之后作用到注意力分数上。

## 接口定义

```cpp
aclnnStatus aclnnHstuForwardV2GetWorkspaceSize(
    aclTensor* q,
    aclTensor* k,
    aclTensor* v,
    aclTensor* mask,
    aclTensor* rab,
    aclTensor* seqOffsetQ,
    aclTensor* seqOffsetK,
    aclTensor* numContext,
    aclTensor* numTarget,
    aclTensor* metadata,
    aclTensor* outputAccum,
    aclTensorList* sparseInfo,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenK,
    double siluScale,
    int64_t targetGroupSize,
    double alpha,
    int64_t maskType,
    int64_t windowSizeLeft,
    int64_t windowSizeRight,
    bool deterministic,
    aclTensor* outAttnOutput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
);

aclnnStatus aclnnHstuForwardV2(
    void* workspaceAddr,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream
);
```

## 参数说明

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **mask** (*Tensor, optional*) – 自定义 mask 张量，`maskType=3` 时必须提供。乘性 0/1 掩码，形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`，可为空指针

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, 1|heads, maxSeqLenQ, maxSeqLenK]`，head 为 1 时广播，可为空指针

- **seqOffsetQ** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **seqOffsetK** (*Tensor, optional*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`。可为空指针（复用 `seqOffsetQ`）

- **numContext** (*Tensor, optional*) – 每个 batch 的 Context 长度。形状: `[batchSize]`，数据类型: `int32`。可为空指针

- **numTarget** (*Tensor, optional*) – 每个 batch 的 Target 长度。形状: `[batchSize]`，数据类型: `int32`。可为空指针

- **metadata** (*Tensor, optional*) – 预生成的分核 metadata（int32，HEAD+FA+FD 布局）。可为空指针（kernel 内现算分核）

- **outputAccum** (*Tensor, optional*) – 非确定性累加缓冲，形状: `[heads, totalSeqLenQ, dimGV]`，数据类型: `float32`。仅 `deterministic=false` 时需要，`deterministic=true` 时传空指针

- **sparseInfo** (*TensorList, optional*) – 任意 mask 的压缩表示。必须恰好包含 9 个 int32 张量。可为空列表

- **maxSeqLenQ** (*int*) – Query 的最大序列长度

- **maxSeqLenK** (*int*) – Key/Value 的最大序列长度

- **siluScale** (*float*) – SiLU 之后的缩放因子。取 `0` 时实际使用 `1.0 / maxSeqLenQ`

- **targetGroupSize** (*int*) – RAG 目标组大小。默认值: `0`

- **alpha** (*float*) – Alpha 系数，作用于 `QK^T + rab`。必须大于 `0`

- **maskType** (*int*) – mask 类型。`0`: 因果掩码（下三角）；`1`: 上三角，不支持；`2`: 无掩码；`3`: 自定义 mask（需提供 `mask` 张量）

- **windowSizeLeft** (*int*) – 注意力窗口左侧宽度，默认 `-1`。`-1` 表示向左无限延伸。与 `windowSizeRight` 共同决定 mask 类型

- **windowSizeRight** (*int*) – 注意力窗口右侧宽度，默认 `-1`。`-1` 表示向右无限延伸，`0` 表示因果掩码。与 `windowSizeLeft` 共同决定 mask 类型

- **deterministic** (*bool*) – 是否确定性累加。默认值: `true`。设为 `false` 时采用 Stream-K 非确定性归约，性能更优但结果可能存在微小误差

- **outAttnOutput** (*Tensor*) – 输出张量。形状: `[totalSeqLenQ, heads, dimGV]`，数据类型与 `q` 一致

## 支持的数据类型

`float16, bfloat16`（q/k/v/mask/rab 与输出一致）；`seqOffsetQ/seqOffsetK/numContext/numTarget`: `int32`；`metadata`: `int32`；`outputAccum`: `float32`

## Shape

- **输入**:
    - `q`: `(totalSeqLenQ, heads, dimQK)`
    - `k`: `(totalSeqLenK, heads, dimQK)`
    - `v`: `(totalSeqLenK, heads, dimGV)`
    - `mask` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`
    - `rab` (optional): `(batchSize, 1|heads, maxSeqLenQ, maxSeqLenK)`
    - `seqOffsetQ`: `(batchSize + 1)`
    - `seqOffsetK` (optional): `(batchSize + 1)`
    - `numContext`/`numTarget` (optional): `(batchSize)`
    - `outputAccum` (optional): `(heads, totalSeqLenQ, dimGV)`

- **输出**:
    - `outAttnOutput`: `(totalSeqLenQ, heads, dimGV)`，dtype 与 `q` 一致

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型，`q/k/v` 必须为 3D 且 dtype 一致
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数且 ≤ 256
- `seqOffsetQ` 必须为 1D `int32` 张量，长度 `batchSize + 1`；`seqOffsetK`（如提供）长度必须与 `seqOffsetQ` 一致
- `batchSize` 必须大于 `0`，`maxSeqLenQ` 与 `maxSeqLenK` 必须大于 `0`
- `k` 的 `heads/dimQK` 必须与 `q` 一致；`v` 的行数必须与 `k` 一致、`heads` 必须与 `q` 一致
- `maskType` 取值范围 `[0, 3]`，不支持 `1`（triu）；`maskType=3` 时必须提供 `mask` 张量
- `windowSizeLeft` 和 `windowSizeRight` 必须 ≥ `-1` 且 `< maxSeqLenK`
- 非局部窗口（`windowSizeLeft = -1` 且 `windowSizeRight ≤ 0`）时，`maskType` 只能为 `0`（causal）或 `2`（none）
- 当提供 `numContext` 或 `numTarget` 时，必须使用 causal 或 local mask（与 backward 一致）
- `alpha` 必须大于 `0`
- `sparseInfo` 提供时必须恰好包含 9 个张量

## mask 行为说明

通过 `maskType` 与 `windowSizeLeft`、`windowSizeRight` 共同控制注意力 mask 类型：

| maskType | 含义 | 说明 |
|----------|------|------|
| 0 | causal mask（因果掩码） | 下三角掩码，每个 token 只能关注自身及之前的 token |
| 1 | triu | 不支持 |
| 2 | no mask（无掩码） | 不做任何 mask，每个 token 可关注所有 token |
| 3 | custom mask | 乘性自定义 mask，必须提供形状 `[batchSize, heads, maxSeqLenQ, maxSeqLenK]` 的 mask 张量 |

通过 `windowSizeLeft` 和 `windowSizeRight` 还可实现局部窗口（local window）注意力：`windowSizeLeft ≥ 0` 或 `windowSizeRight > 0` 时，query 仅关注左侧 `windowSizeLeft` 和右侧 `windowSizeRight` 范围。

当使用 causal mask 时，可通过 `numContext` 和 `numTarget` 进一步细分 mask 区域：

- **context 区域**（`numContext` 指定长度）：双向注意力，无 causal 限制
- **target 区域**（`numTarget` 指定长度）：分组注意力，group_size 由 `targetGroupSize` 控制
- **其余区域**：标准 causal mask（下三角）

任意 mask 亦可不传入稠密 mask 张量，而是通过 `get_fwd_attn_plan` 生成的 `sparseInfo`（9 个打包张量）压缩表示传入。

注意：`maskType` 与 `windowSize` 的合法组合见「约束条件」；`numContext`/`numTarget` 仅用于 causal 场景，不与自定义 mask 张量组合。

## 调用示例

调用流程（GetWorkspaceSize → 分配 workspace → 执行 → 同步）与 backward 示例 [test_aclnn_hstu_backward_v2.cpp](../examples/aclnn/test_aclnn_hstu_backward_v2.cpp) 一致；参数顺序以 `torch_extension/hstu_forward_v2_plugin.cpp` 中的 `EXEC_NPU_CMD(aclnnHstuForwardV2, ...)` 为准：

```cpp
// q, k, v, mask(可空), rab(可空), seqOffsetQ, seqOffsetK(可空),
// numContext(可空), numTarget(可空), metadata(可空), outputAccum(可空),
// sparseInfo(可空列表) 为输入；
// maxSeqLenQ, maxSeqLenK, siluScale, targetGroupSize, alpha,
// maskType, windowSizeLeft, windowSizeRight, deterministic 为属性；
// outAttnOutput 为输出。
ret = aclnnHstuForwardV2GetWorkspaceSize(
    q, k, v, nullptr, rab, seqOffsetQ, seqOffsetK, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    maxSeqLenQ, maxSeqLenK, 1.0 / static_cast<double>(maxSeqLenQ),
    1, 1.0, 0, -1, 0, true,
    attnOutput, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuForwardV2GetWorkspaceSize failed. ERROR: %d\n", ret);
          return ret);

void* workspaceAddr = nullptr;
if (workspaceSize != 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
}

ret = aclnnHstuForwardV2(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuForwardV2 failed. ERROR: %d\n", ret); return ret);

ret = aclrtSynchronizeDevice();
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);
```
