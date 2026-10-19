# aclnnBuildGistMaskArbitraryFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据 GIST（Generalized Instruction-aware Sparse Transformer）规则构造 arbitrary function 掩码编码（af 编码）：对每个全局位置 `pos`，
二分查找 `offsets` 定位所属序列 `seq_id`，根据 query 位置落在 pre_doc / gist / 其他区间的不同，写出
`af[pos][0..3]` 四个列（全局坐标，最多两个可见区间 `[col0, col1) ∪ [col2, col3)`，以空区间填充）。

设 `seq_start=offsets[seq_id]`、`seq_end=offsets[seq_id+1]`、`pos_local=pos-seq_start`，
`p=preDocLens[seq_id]`、`g0=gistStarts[seq_id]`、`g1=gistEnds[seq_id]`（均已 clamp 到 `[0, seq_len]`）：

- pre_doc token（`pos_local < p`）：可见 `[seq_start, pos+1)`
  → `col0=seq_start, col1=pos+1, col2=col3=seq_start`
- 中间 token（`p ≤ pos_local < g0`）：可见 `[seq_start, seq_start+p) ∪ [seq_start+p, pos+1)`
  → `col0=seq_start, col1=seq_start+p, col2=seq_start+p, col3=pos+1`
- gist token（`g0 ≤ pos_local < g1`）：可见 `[seq_start, pos+1)`
  → `col0=seq_start, col1=pos+1, col2=col3=seq_start`
- gist 之后 token（`pos_local ≥ g1`）：可见 `[seq_start, seq_start+p) ∪ [seq_start+g0, seq_start+g1)`
  → `col0=seq_start, col1=seq_start+p, col2=seq_start+g0, col3=seq_start+g1`

输出 shape `[1, 1, total_seqlen, 4]`，每个 query 可见 key 集合 = `[col0, col1) ∪ [col2, col3)`（全局索引）。

### 编码示意图

![build_gist_mask_arbitrary_func af编码与注意力mask对应关系](build_gist_mask_arbitrary_func示意图.svg)

## 函数原型

两段式接口，必须先调用 `aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize` 接口获取 workspace 大小和执行器，
再调用 `aclnnBuildGistMaskArbitraryFunc` 接口执行计算。

```c
aclnnStatus aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize(
    const aclTensor *offsets,
    const aclTensor *preDocLens,
    const aclTensor *gistStarts,
    const aclTensor *gistEnds,
    int64_t totalSeqLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBuildGistMaskArbitraryFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| offsets（aclTensor） | 输入 | 每序列起始偏移，shape [batch_size+1] | INT32 | ND |
| preDocLens（aclTensor） | 输入 | 每序列 pre_doc 长度，shape [batch_size] | INT32 | ND |
| gistStarts（aclTensor） | 输入 | 每序列 gist 起始位置，shape [batch_size] | INT32 | ND |
| gistEnds（aclTensor） | 输入 | 每序列 gist 结束位置，shape [batch_size] | INT32 | ND |
| totalSeqLen（int64_t） | 输入 | 所有序列长度之和，须等于 `offsets` 最后一个元素 | INT64 | 标量 |
| out（aclTensor） | 输出 | af 编码输出，shape [1,1,total_seqlen,4] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | UINT64 | 指针 |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | 执行器指针 | 指针 |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnBuildGistMaskArbitraryFunc

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D, `offsets[0]` 必须为 0、`offsets` 严格单调递增
- `preDocLens`、`gistStarts`、`gistEnds` 必须为 1D，shape 须为 `[batch_size]`（batch_size = offsets.shape[0] - 1）
- `totalSeqLen` 须等于 `offsets[-1]`
- `preDocLens[i]`、`gistStarts[i]`、`gistEnds[i]` 须在 `[0, seq_len]` 范围内（算子内部会 clamp）
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_arbitrary_func_gist_mask.cpp](../examples/aclnn/test_aclnn_arbitrary_func_gist_mask.cpp) 的完整实现。
