# build_hstu_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |

## 功能说明

根据每个序列的 `offsets` / `ctx_lens` / `tgt_lens`，构造 arbitrary function 掩码编码（af 编码）。

对每个全局位置 `pos`，二分查找 `offsets` 定位所属序列 `seq_id`，再按 `pos_in_seq` 与
`ctx_len` / `hist_end` 三段判断 ctx / history / target，写出 `af[pos][0..3]` 四个列：

- context token（`pos_in_seq < ctx_len`）：`col0=seq_start, col1=base, col2=0, col3=0`
- history token（`pos_in_seq < hist_end`）：`col0=seq_start, col1=pos+1, col2=0, col3=0`
- target token（否则）：记 `tgt_idx=pos_in_seq-hist_end`，
  `group_start=(tgt_idx/target_group_size)*target_group_size`：
  `col0=seq_start, col1=base, col2=base+group_start, col3=base+tgt_idx+1`

其中 `seq_start=offsets[seq_id]`，`base=seq_start+hist_end`，`hist_end=ctx_len+hist_len`，
`hist_len=seq_len-ctx_len-tgt_len`。

### 编码示意图

![build_hstu_mask_arbitrary_func af编码与注意力mask对应关系](build_hstu_mask_arbitrary_func示意图.svg)

## 接口原型

```python
ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(offsets, ctx_lens, tgt_lens, target_group_size, total_seqlen) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| offsets | 输入 | Tensor | INT32 | ND | [batch_size+1] | 每序列起始偏移 |
| ctx_lens | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 context 长度 |
| tgt_lens | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 target 长度 |
| target_group_size | 输入 | INT64 | - | - | - | target 分组的组大小标量 |
| total_seqlen | 输入 | INT64 | - | - | - | 所有序列长度之和 |
| output | 输出 | Tensor | INT32 | ND | [1, 1, total_seqlen, 4] | af 编码输出 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用
`ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func` 时生效。

### 算子约束

- 仅支持 INT32 数据类型
- `offsets`、`ctx_lens`、`tgt_lens` 必须为 1D
- `ctx_lens` / `tgt_lens` 的维度须满足 `[batch_size]` / `[batch_size]`（`batch_size = offsets.size(0)-1`），不满足时算子直接报错返回
- `total_seqlen` 须等于 `offsets[-1]`
- 该算子需要在 NPU 设备上运行
- `target_group_size` 须大于 0

### torch 扩展层额外校验

由 [arbitrary_func_plugin.cpp](../torch_extension/arbitrary_func_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- `ctx_lens` / `tgt_lens` 的 shape 须满足 `[batch_size]` / `[batch_size]`（`batch_size = offsets.size(0)-1`）
- `total_seqlen` 必须等于 `offsets[-1]`（读取 device 数据显式校验）
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install
```

## 调用示例

完整调用示例参考 [test_torch_arbitrary_func_hstu_mask.py](../examples/torch/test_torch_arbitrary_func_hstu_mask.py)。
