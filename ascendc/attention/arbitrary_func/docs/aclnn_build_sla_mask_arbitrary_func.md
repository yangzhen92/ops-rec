# aclnnBuildSlaMaskArbitraryFunc

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | × |
| Atlas A2  训练/推理系列产品 | × |

## 功能说明

根据 SLA（Semi-Local Attention）规则构造 arbitrary function 掩码编码（af 编码）：对每个全局位置 `pos`，
二分查找 `offsets` 定位所属序列 `seq_id`，计算局部位置与 history 边界后按 history / target 情况写出
`af[pos][0..3]` 四个列（全局坐标，半开区间 `[col0, col1) ∪ [col2, col3)`，`col1 ≤ col0 时第一区间为空、col3 ≤ col2 时第二区间为空`）。

设 `seq_start=offsets[seq_id]`、`seq_len=offsets[seq_id+1]-seq_start`、`pos_local=pos-seq_start`，
`eff_k2=max(sla_k2, contextual_seq_len)`、`H=max(seq_len-num_targets[seq_id], 0)`：

- history token（`pos_local < H`）：
  `c0=min(pos_local+1, eff_k2)`，`c1=max(eff_k2, pos_local-sla_k1+1)`，`c2=pos_local+1`
  → `col0=seq_start, col1=seq_start+c0, col2=seq_start+c1, col3=seq_start+c2`
- target token（`pos_local ≥ H`）：**只见完整 history `[0, H)`，连自身也不可见**
  `c0=c1=c2=H` → `col0=col1=col2=col3=seq_start+H`

输出 shape `[1, 1, total_seqlen, 4]`，每个 query 可见 key 集合 = `[col0, col1) ∪ [col2, col3)`（全局索引）。

### 编码示意图

![build_sla_mask_arbitrary_func af编码与注意力mask对应关系](build_sla_mask_arbitrary_func示意图.svg)

## 函数原型

两段式接口，必须先调用 `aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize` 接口获取 workspace 大小和执行器，
再调用 `aclnnBuildSlaMaskArbitraryFunc` 接口执行计算。

```c
aclnnStatus aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize(
    const aclTensor *offsets,
    const aclTensor *numTargets,
    int64_t slaK1,
    int64_t slaK2,
    int64_t contextualSeqLen,
    int64_t totalSeqLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBuildSlaMaskArbitraryFunc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| offsets（aclTensor） | 输入 | 每序列起始偏移，shape [batch_size+1] | INT32 | ND |
| numTargets（aclTensor） | 输入 | 每序列 target 数，shape [batch_size]（须等于 `offsets.size(0)-1`；缺省语义为全 0，需调用方自行构造） | INT32 | ND |
| slaK1（int64_t） | 输入 | Local 窗口大小标量 | INT64 | 标量 |
| slaK2（int64_t） | 输入 | Global 前缀长度标量 | INT64 | 标量 |
| contextualSeqLen（int64_t） | 输入 | context token 数标量，参与 `eff_k2=max(sla_k2, contextual_seq_len)` | INT64 | 标量 |
| totalSeqLen（int64_t） | 输入 | 所有序列长度之和，须等于 `offsets` 最后一个元素 | INT64 | 标量 |
| out（aclTensor） | 输出 | af 编码输出，shape [1,1,total_seqlen,4] | INT32 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | UINT64 | 指针 |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | 执行器指针 | 指针 |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnBuildSlaMaskArbitraryFunc

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D, `offsets[0]` 必须为 0、`offsets` 严格单调递增
- `numTargets` 必须为 1D, `numTargets` shape 须为 `[batch_size]`（batch_size = offsets.shape[0] - 1），长度不匹配会报错
- `totalSeqLen` 须等于 `offsets[-1]`
- `slaK1`、`slaK2`、`contextualSeqLen` 须大于等于 0
- 该算子需要在 NPU 设备上运行
- 语义上 `num_targets` 可缺省（等价于全 0），但本接口要求传入 shape 为 `[batch_size]` 的完整 tensor，
  缺省时调用方应传入全 0 张量（torch 接口层会在用户未传时自动补全 0）

## 调用示例

具体参考 [test_aclnn_arbitrary_func_sla_mask.cpp](../examples/aclnn/test_aclnn_arbitrary_func_sla_mask.cpp) 的完整实现。
