# build_*_mask_arbitrary_func（arbitrary function 掩码构造）

该目录实现两类基于 AscendC SIMT 的 arbitrary function（af）掩码构造算子：对每个全局位置 `pos` 写出
4 个全局索引列 `af[pos][0..3]`，表示该位置可见的 key 集合 `[col0, col1) ∪ [col2, col3)`
（半开区间，越界忽略）。

- [build_hstu_mask_arbitrary_func](#build_hstu_mask_arbitrary_func)：HSTU 掩码，按 ctx / history / target 三段划分
- [build_sla_mask_arbitrary_func](#build_sla_mask_arbitrary_func)：SLA 掩码，按 history / target 两段划分

---

# build_hstu_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3 系列产品 | × |
| Atlas A2 系列产品 | × |

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

---

# build_sla_mask_arbitrary_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3 系列产品 | × |
| Atlas A2 系列产品 | × |

## 功能说明

根据 SLA（Semi-Local Attention）规则构造 arbitrary function 掩码编码（af 编码）：对每个全局位置 `pos`，
二分查找 `offsets` 定位所属序列 `seq_id`，按 history / target 两段判断后写出 `af[pos][0..3]` 四个列
（全局坐标，半开区间 `[col0, col1) ∪ [col2, col3)`，`col1 ≤ col0` 时第一区间为空、`col3 ≤ col2` 时第二区间为空）。

设 `seq_start=offsets[seq_id]`，`seq_len=offsets[seq_id+1]-seq_start`，`pos_local=pos-seq_start`，
`eff_k2=max(sla_k2, contextual_seq_len)`，`H=max(seq_len-num_targets[seq_id], 0)`：

- history token（`pos_local < H`）：
  `c0=min(pos_local+1, eff_k2)`，`c1=max(eff_k2, pos_local-sla_k1+1)`，`c2=pos_local+1`
  → `col0=seq_start, col1=seq_start+c0, col2=seq_start+c1, col3=seq_start+c2`
- target token（`pos_local ≥ H`）：**只见完整 history `[0, H)`，连自身也不可见**（两区间均为空）
  `c0=c1=c2=H` → `col0=col1=col2=col3=seq_start+H`

其中第一个 interval 为 global 前缀（取 `eff_k2` 与当前位置的较小者），第二个 interval 为 sliding local 窗口。
每个 query 可见 key 集合 = `[col0, col1) ∪ [col2, col3)`（全局索引，越界忽略）。

对应接口：

- aclnn 接口：`aclnnBuildSlaMaskArbitraryFunc`
- torch 接口：`ops_rec.ascendc.attention.build_sla_mask_arbitrary_func`

## 算子实现原理

算子采用 AscendC SIMT（`__simt_vf__`）开发。每个全局 token 位置由一个 SIMT 线程处理，
核心步骤：

1. 二分查找定位 `pos` 所属序列 `seq_id`
2. 计算 `seq_start` / `H` / `eff_k2`
3. 按 history / target 两段分类，显式写出 4 列：
   - history：`c0=min(pos_local+1, eff_k2)`、`c1=max(eff_k2, pos_local-sla_k1+1)`、`c2=pos_local+1`
   - target：`c0=c1=c2=H`（只见完整 history，连自身也不可见）
4. 用单次 int4（16B）向量 store 写出 4 列

SIMT 内核由 `op_kernel/build_sla_mask_arbitrary_func_kernel.h` 的 `SimtBuildSlaMaskArbitraryFunc` 实现，经
`extern "C" __global__ __aicore__` 入口 `VF_CALL` 调度。

与 HSTU 掩码的差异：SLA 仅需 `offsets` 与 `num_targets` 两个 tensor 输入（`H=seq_len-num_targets`，history
长度由序列总长推导，无需单独的 `hist_len`），划分维度从 ctx / history / target 三段简化为 history / target
两段；`eff_k2=max(sla_k2, contextual_seq_len)` 在循环外提前计算。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| offsets | 输入 | Tensor | INT32 | ND | [batch_size+1] | 每序列起始偏移（cu_seqlens） |
| num_targets | 输入 | Tensor | INT32 | ND | [batch_size] | 每序列 target 数，`H=seq_len-num_targets`；语义可缺省（全 0） |
| sla_k1 | 输入 | int | - | - | - | Local 窗口大小标量 |
| sla_k2 | 输入 | int | - | - | - | Global 前缀长度标量 |
| contextual_seq_len | 输入 | int | - | - | - | context token 数，参与 `eff_k2=max(sla_k2, contextual_seq_len)` |
| total_seqlen | 输入 | int | - | - | - | 所有序列长度之和 |
| output | 输出 | Tensor | INT32 | ND | [1, 1, total_seqlen, 4] | af 编码输出 |

## 约束说明

- 仅支持 INT32 数据类型
- `offsets` 必须为 1D，`offsets[0]` 须为 0、`offsets` 严格单调递增
- `num_targets` 必须为 1D，shape 须为 `[batch_size]`（`batch_size = offsets.size(0)-1`），不满足时算子直接报错返回
- `total_seqlen` 须等于 `offsets[-1]`
- `sla_k1`、`sla_k2`、`contextual_seq_len` 须大于等于 0
- 该算子需要在 NPU 设备上运行

> `num_targets` 语义上可缺省（等价于全 0，即全部为 history token）；aclnn 接口要求传入 shape 为
> `[batch_size]` 的完整 tensor，缺省时调用方应传入全 0 张量；torch 接口未传时由扩展层自动以全 0 补齐。
> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 必须已定义且非空、所有输入 tensor 必须位于
> 同一 NPU 设备、`num_targets` 传入时 shape 须为 `[batch_size]`、`total_seqlen` 等于 `offsets[-1]`。
> 详见 [build_sla_mask_arbitrary_func 接口文档](./docs/torch_build_sla_mask_arbitrary_func.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_arbitrary_func_sla_mask.cpp)
- [torch 调用示例](./examples/torch/test_torch_arbitrary_func_sla_mask.py)
- [aclnn 接口文档](./docs/aclnn_build_sla_mask_arbitrary_func.md)
- [torch 接口文档](./docs/torch_build_sla_mask_arbitrary_func.md)
