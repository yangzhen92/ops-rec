# build_gist_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据 GIST（Generalized Instruction-aware Sparse Transformer）规则构造 arbitrary function 掩码编码（af 编码）：对每个全局位置 `pos`，
二分查找 `offsets` 定位所属序列 `seq_id`，根据 query 位置落在 pre_doc / gist / 其他区间的不同，写出
`af[pos][0..3]` 四个列（全局坐标，最多两个可见区间 `[col0, col1) ∪ [col2, col3)`）。

设 `seq_start=offsets[seq_id]`、`seq_end=offsets[seq_id+1]`、`pos_local=pos-seq_start`，
`p=pre_doc_lens[seq_id]`、`g0=gistStarts[seq_id]`、`g1=gistEnds[seq_id]`：

- pre_doc token（`pos_local < p`）：可见 `[seq_start, pos+1)`
  → `col0=seq_start, col1=pos+1, col2=col3=seq_start`
- 中间 token（`p ≤ pos_local < g0`）：可见 `[seq_start, seq_start+p) ∪ [seq_start+p, pos+1)`
  → `col0=seq_start, col1=seq_start+p, col2=seq_start+p, col3=pos+1`
- gist token（`g0 ≤ pos_local < g1`）：可见 `[seq_start, pos+1)`
  → `col0=seq_start, col1=pos+1, col2=col3=seq_start`
- gist 之后 token（`pos_local ≥ g1`）：可见 `[seq_start, seq_start+p) ∪ [seq_start+g0, seq_start+g1)`
  → `col0=seq_start, col1=seq_start+p, col2=seq_start+g0, col3=seq_start+g1`

输出 shape `[1, 1, total_seqlen, 4]`，head 维恒 1（所有 head 广播共享同一 mask）。

### 编码示意图

![build_gist_mask_arbitrary_func af编码与注意力mask对应关系](build_gist_mask_arbitrary_func示意图.svg)

## 接口原型

```python
ops_rec.ascendc.attention.build_gist_mask_arbitrary_func(
    offsets, pre_doc_lens, gist_starts, gist_ends, total_seqlen) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| offsets | 输入 | Tensor | INT32 | ND | [batch_size+1] | 每序列起始偏移（cu_seqlens） |
| pre_doc_lens | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 pre_doc token 数 |
| gist_starts | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 gist 起始位置 |
| gist_ends | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 gist 结束位置 |
| total_seqlen | 输入 | INT64 | - | - | 标量 | 所有序列长度之和，须等于 `offsets[-1]` |
| output | 输出 | Tensor | INT32 | ND | [1, 1, total_seqlen, 4] | af 编码输出 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用
`ops_rec.ascendc.attention.build_gist_mask_arbitrary_func` 时生效。

### 算子约束

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D, `offsets[0]` 必须为 0、`offsets` 严格单调递增
- `preDocLens`、`gistStarts`、`gistEnds` 必须为 1D，shape 须为 `[batch_size]`
- `totalSeqLen` 须等于 `offsets[-1]`
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [arbitrary_func_plugin.cpp](../torch_extension/arbitrary_func_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- 需要安装 `torch_npu`
- `total_seqlen` 必须等于 `offsets[-1]`（总 token 数），否则报错
- `pre_doc_lens`、`gist_starts`、`gist_ends` 的 shape 必须为 `[batch_size]`（= `offsets.size(0)-1`），否则报错

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_arbitrary_func_gist_mask.py](../examples/torch/test_torch_arbitrary_func_gist_mask.py)。
