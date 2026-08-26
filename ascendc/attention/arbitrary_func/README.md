# build_hstu_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | × |
| Atlas A2  训练/推理系列产品 | × |

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

对应接口：

- aclnn 接口：`aclnnBuildHstuMaskArbitraryFunc`
- torch 接口：`ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func`

## 算子实现原理

算子采用 AscendC SIMT（`__simt_vf__`）开发。每个全局 token 位置由一个 SIMT 线程处理，
核心步骤：

1. 二分查找定位 `pos` 所属序列 `seq_id`
2. 计算 `seq_start` / `base` / `hist_end`
3. 按 ctx / history / target 三段分类，显式写出 4 列（非 target 的 col2/col3 填 0）
4. 用单次 int4（16B）向量 store 写出 4 列

SIMT 内核由 `op_kernel/build_hstu_mask_arbitrary_func_kernel.h` 的 `SimtBuildHstuMaskArbitraryFunc` 实现，经
`extern "C" __global__ __aicore__` 入口 `VF_CALL` 调度。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| offsets | 输入 | Tensor | INT32 | ND | [batch_size+1] | 每序列起始偏移 |
| ctx_lens | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 context 长度 |
| tgt_lens | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 target 长度 |
| target_group_size | 输入 | int | - | - | - | target 分组的组大小标量 |
| total_seqlen | 输入 | int | - | - | - | 所有序列长度之和 |
| output | 输出 | Tensor | INT32 | ND | [1, 1, total_seqlen, 4] | af 编码输出 |

## 约束说明

- 仅支持 INT32 数据类型
- `offsets`、`ctx_lens`、`tgt_lens` 必须为 1D，且维度须满足 `[batch_size+1]` / `[batch_size]` / `[batch_size]`（`batch_size = offsets.size(0)-1`），不满足时算子直接报错返回
- `total_seqlen` 须等于 `offsets[-1]`
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 必须已定义且非空、所有输入 tensor 必须位于同一 NPU 设备、三个输入 shape 满足 `[batch_size+1]` / `[batch_size]` / `[batch_size]`、`total_seqlen` 等于 `offsets[-1]`。详见 [build_hstu_mask_arbitrary_func 接口文档](./docs/torch_build_hstu_mask_arbitrary_func.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_arbitrary_func_hstu_mask.cpp)
- [torch 调用示例](./examples/torch/test_torch_arbitrary_func_hstu_mask.py)
- [aclnn 接口文档](./docs/aclnn_build_hstu_mask_arbitrary_func.md)
- [torch 接口文档](./docs/torch_build_hstu_mask_arbitrary_func.md)
