# build_causal_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据每个序列的 `offsets` 构造 causal（因果）掩码的 arbitrary function 掩码编码（af 编码）：对每个全局位置
`pos`，二分查找 `offsets` 定位所属序列 `seq_id`，写出 `af[pos][0..1]` 两个列（全局坐标，半开区间 `[col0, col1)`，
`col1 ≤ col0` 视为空区间）：

- `col0=seq_start=offsets[seq_id]`
- `col1=global_pos+1`（含自身）

可见 key 集合 = `[col0, col1)`，即当前序列内不晚于当前 query 的所有 token（`k <= q`）。

输出 shape `[1, 1, total_seqlen, 2]`（单区间 mask，`N_FUNC=2`），head 维恒 1（所有 head 广播共享同一 mask）。

### 编码示意图

![build_causal_mask_arbitrary_func af编码与注意力mask对应关系](build_causal_mask_arbitrary_func示意图.svg)

## 接口原型

```python
ops_rec.ascendc.attention.build_causal_mask_arbitrary_func(offsets, total_seqlen) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| offsets | 输入 | Tensor | INT32 | ND | [batch_size+1] | 每序列起始偏移（cu_seqlens） |
| total_seqlen | 输入 | INT64 | - | - | 标量 | 所有序列长度之和，须等于 `offsets[-1]` |
| output | 输出 | Tensor | INT32 | ND | [1, 1, total_seqlen, 2] | af 编码输出 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用
`ops_rec.ascendc.attention.build_causal_mask_arbitrary_func` 时生效。

### 算子约束

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D, `offsets[0]` 必须为 0、`offsets` 严格单调递增
- `total_seqlen` 须等于 `offsets[-1]`
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [arbitrary_func_plugin.cpp](../torch_extension/arbitrary_func_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- 需要安装 `torch_npu`
- `offsets` 必须为 1D 的 int32 tensor
- `total_seqlen` 须满足 `0 <= total_seqlen <= INT32_MAX`
- `total_seqlen` 必须等于 `offsets[-1]`（总 token 数），否则报错

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_arbitrary_func_causal_mask.py](../examples/torch/test_torch_arbitrary_func_causal_mask.py)。
