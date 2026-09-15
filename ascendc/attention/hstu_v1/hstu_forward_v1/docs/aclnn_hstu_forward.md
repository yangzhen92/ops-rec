# aclnnHstuDenseForward / aclnnHstuJaggedForward / aclnnHstuPagedForward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

HSTU（Hierarchical Sequential Transduction Unit）融合注意力算子，支持 normal（dense）、jagged、paged 三种数据格式，支持 GQA（K/V 头数小于 Q 头数）与 qk/v dim 不等。

计算公式：

```text
HSTU(q, k, v, mask, bias, siluScale) = (Silu(qk^T + bias) * siluScale * mask) v
```

三个接口对应三种数据格式：

| 接口 | 数据格式 | 输入 q/k/v shape |
|------|----------|------------------|
| aclnnHstuDenseForward | normal | [B, S, N, D] |
| aclnnHstuJaggedForward | jagged | [s_b, N, D] |
| aclnnHstuPagedForward | paged | [s_b, N, D]（配合 kv_cache 分页） |

## 函数原型

两段式接口，必须先调用 `aclnnHstuXxxForwardGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnHstuXxxForward` 接口执行计算。

### aclnnHstuDenseForwardGetWorkspaceSize

```c
aclnnStatus aclnnHstuDenseForwardGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *mask, const aclTensor *attnBias,
    int64_t maskType, int64_t maxSeqLen, double siluScale,
    const aclTensor *attnOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHstuDenseForward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

### aclnnHstuJaggedForwardGetWorkspaceSize

```c
aclnnStatus aclnnHstuJaggedForwardGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *mask, const aclTensor *attnBias,
    const aclTensor *seqOffsetQ, const aclTensor *seqOffsetK,
    const aclTensor *numContext, const aclTensor *numTarget,
    int64_t maskType, int64_t maxSeqLenQ, int64_t maxSeqLenK, double siluScale,
    int64_t targetGroupSize, double alpha, bool deterministic,
    const aclTensor *attnOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHstuJaggedForward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

### aclnnHstuPagedForwardGetWorkspaceSize

```c
aclnnStatus aclnnHstuPagedForwardGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *mask, const aclTensor *attnBias,
    const aclTensor *seqOffsetQ, const aclTensor *seqOffsetK, const aclTensor *seqOffsetT,
    const aclTensor *kvCache, const aclTensor *pageOffsets, const aclTensor *pageIds,
    const aclTensor *lastPageLen, const aclTensor *numContext, const aclTensor *numTarget,
    int64_t maskType, int64_t maxSeqLenQ, int64_t maxSeqLenK, double siluScale,
    int64_t targetGroupSize, double alpha, bool deterministic,
    const aclTensor *attnOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHstuPagedForward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

> 注：可选入参（mask、attn_bias、num_context、num_target 等）不使用时传入 `nullptr`；siluScale 传 0.0 时自动取 `1/maxSeqLen`。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| q（aclTensor） | 输入 | Query 张量，[B, S, N_q, D_q] 或 [s_b, N_q, D_q] | FLOAT32/FLOAT16/BFLOAT16（Ascend 950PR&950DT 系列产品 另支持 FP8_E4M3FN） | ND |
| k（aclTensor） | 输入 | Key 张量，头数 N_k ≤ N_q | 同 q | ND |
| v（aclTensor） | 输入 | Value 张量，head_dim 可与 qk 不等 | 同 q | ND |
| mask（aclTensor） | 输入 | 注意力 mask，[B, N, S, S]（可选） | 同 q | ND |
| attnBias（aclTensor） | 输入 | 相对位置偏置，[B, N, S, S]（可选） | 同 q | ND |
| seqOffsetQ/K/T（aclTensor） | 输入 | 序列偏移量（jagged/paged 必选） | INT32/INT64 | ND |
| kvCache（aclTensor） | 输入 | KV 缓存，[num_pages, 2, page_size, N, D]（paged） | 同 q | ND |
| pageOffsets/pageIds/lastPageLen（aclTensor） | 输入 | 分页元数据（paged） | INT32/INT64 | ND |
| numContext/numTarget（aclTensor） | 输入 | 上下文/目标数量（可选） | INT32/INT64 | ND |
| maskType（int64_t） | 输入 | 0：下三角；2：不使用；3：自定义（1 暂不支持） | - | - |
| maxSeqLenQ/maxSeqLenK（int64_t） | 输入 | 序列最大长度，[1, 20480] | - | - |
| siluScale（double） | 输入 | Silu 缩放，默认 1/maxSeqLen | - | - |
| targetGroupSize（int64_t） | 输入 | 内置 target mask 分组大小（jagged/paged） | - | - |
| alpha（double） | 输入 | Alpha 缩放参数 | - | - |
| deterministic（bool） | 输入 | 确定性计算开关 | - | - |
| attnOutput（aclTensor） | 输出 | 输出张量，[B, S, N_q, D_v] 或 [s_b, N_q, D_v] | 同 q（Ascend 950PR&950DT 系列产品 FP8 输入时为 FLOAT16） | ND |
| workspaceSize（uint64_t*） | 输出 | 需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 算子执行器 | - | - |

## 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## 约束说明

- q/k/v 的 B、S、N、D 均不能为 0
- GQA：`N_q % N_k == 0` 且 `N_k == N_v`
- v_dim 需为 16 的倍数，范围 [16, 512]
- max_seq_len_q + max_seq_len_k 需满足内存约束
- FP8（fp8_e4m3fn）仅在 Ascend 950PR&950DT 系列产品 上支持，且输出为 float16

## 调用示例

见 [examples/aclnn/](../examples/aclnn/) 目录下的独立调用示例：

- `test_aclnn_hstu_forward_dense.cpp`：aclnnHstuDenseForward（normal 格式）
- `test_aclnn_hstu_forward_jagged.cpp`：aclnnHstuJaggedForward（jagged 格式，delta 语义）
- `test_aclnn_hstu_forward_paged.cpp`：aclnnHstuPagedForward（paged 格式）
- `test_aclnn_hstu_forward_common.h`：三个示例共用的公共代码（宏 / Init / CreateAclTensor）
