# create_attn_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |

## 功能说明

根据序列长度、注意力头数、掩码及分块信息，在 AI CPU 上生成注意力计算的调度元数据。
输出记录分块大小和各核的任务范围，供 HSTU 等注意力算子的 metadata 调度分支使用。
本算子不执行 Q、K、V 的注意力计算，也不生成 packed 块索引。

前向按 Q 行分配任务，反向按 K 行分配任务。提供 packed 块计数时，可根据每行
MASK 块和 FULL 块的数量估算负载。生成与消费 metadata 时，序列、方向、分块大小、
确定性模式及运行设备必须保持一致。

## 接口原型

```python
ops_rec.ascendc.attention.create_attn_metadata(
    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
    batch_size, max_seqlen_q, max_seqlen_kv,
    num_heads_q, num_heads_kv, head_dim,
    mask_mode, win_left, win_right,
    layout_q, layout_kv, layout_out,
    mask_block_cnt=None, full_block_cnt=None,
    packed_row_axis=0, packed_mask_heads=1,
    q_block_size=0, kv_block_size=0, deterministic=True,
) -> Tensor
```

也可通过 `torch.ops.ops_rec.create_attn_metadata(...)` 调用。调用前需 `import ops_rec`
加载并注册算子。除带默认值的参数外，其余参数均需显式传入；可选张量可传 `None`。

## 参数说明

`B` 表示 batch 数，`H` 表示 packed 掩码头数，`outer` 表示 packed 计数表的行容量。

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 / shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| cu_seqlens_q | 输入 | Tensor 或 None | INT32、INT64 | ND / [B+1] | Q 的累积序列偏移，首元素为 0；TND 布局必须提供 |
| cu_seqlens_kv | 输入 | Tensor 或 None | INT32、INT64 | ND / [B+1] | KV 的累积序列偏移，首元素为 0；TND 布局必须提供 |
| seqused_q | 输入 | Tensor 或 None | INT32、INT64 | ND / [B] | 各 batch 的实际 Q 长度；提供时优先于累积偏移用于长度计算 |
| seqused_kv | 输入 | Tensor 或 None | INT32、INT64 | ND / [B] | 各 batch 的实际 KV 长度；提供时优先于累积偏移用于长度计算 |
| batch_size | 输入 | int | — | — | batch 数；大于 0 时直接使用，否则 Torch 层从 seqused_q 或 cu_seqlens_q 推导 |
| max_seqlen_q | 输入 | int | — | — | 最大 Q 序列长度；存在实际长度张量时，内核会根据其内容计算最大长度 |
| max_seqlen_kv | 输入 | int | — | — | 最大 KV 序列长度 |
| num_heads_q | 输入 | int | — | — | Q 头数，必须大于 0 |
| num_heads_kv | 输入 | int | — | — | KV 头数，当前必须等于 num_heads_q |
| head_dim | 输入 | int | — | — | 头维度，取 16 的正整数倍，且不超过 256 |
| mask_mode | 输入 | int | — | — | 0：无掩码；3：因果掩码；4：窗口掩码 |
| win_left | 输入 | int | — | — | 左侧窗口范围，-1 表示不限制，其他取值须非负 |
| win_right | 输入 | int | — | — | 右侧窗口范围，-1 表示不限制，其他取值须非负 |
| layout_q | 输入 | str | — | — | Q 布局：BSND、TND、BNSD |
| layout_kv | 输入 | str | — | — | KV 布局：BSND、TND、BNSD、PA_BNBD、PA_BBND、PA_NZ |
| layout_out | 输入 | str | — | — | 输出布局：BSND、TND、BNSD |
| mask_block_cnt | 输入 | Tensor 或 None | INT32 | ND / [B,H,outer] | 每个外层行块中的 MASK 块数量；默认 None |
| full_block_cnt | 输入 | Tensor 或 None | INT32 | ND / [B,H,outer] | 每个外层行块中的 FULL 块数量；必须与 mask_block_cnt 同时提供且形状一致 |
| packed_row_axis | 输入 | int | — | — | 0：Q 行，前向；1：K 行，反向；默认 0。未提供 packed 计数时也用于选择前反向入口 |
| packed_mask_heads | 输入 | int | — | — | 1 表示所有注意力头共享计数；num_heads_q 表示逐头计数；默认 1 |
| q_block_size | 输入 | int | — | — | Q 方向块大小；默认 0，由内核选择 |
| kv_block_size | 输入 | int | — | — | KV 方向块大小；默认 0，由内核选择 |
| deterministic | 输入 | bool | — | — | 是否采用确定性调度；默认 True；与消费端配置保持一致 |
| metadata | 输出 | Tensor | INT32 | ND / 一维 | NPU 上的调度元数据，容量由 Torch 层分配并按 4096 个 INT32 元素对齐 |

## 约束说明

- 当前 AI CPU 内核仅支持 MHA，即 `num_heads_q == num_heads_kv`，不支持 GQA。
- 张量应位于当前 NPU 设备；Torch 层会将提供的张量转换为连续存储。
- TND 布局必须提供对应的累积偏移；非 TND 布局不能提供对应的累积偏移，且必须
  提供最大序列长度或实际长度张量。累积偏移须单调不减，实际长度须非负。
- packed 计数必须为同形状的三维 INT32 NPU 张量。第一维等于 B，第二维等于
  `packed_mask_heads`；第三维须足以容纳每个 batch 在所选行轴上的所有行块。
  计数张量应来自与消费端相同方向、相同分块配置的 packed 块稀疏结果。
- 指定分块大小时，`q_block_size` 和 `kv_block_size` 应同时为正数；二者任一不为正时，
  当前内核采用默认分块。外层行块由 `packed_row_axis` 决定。
- 提供 packed 计数时，内部选择 `CreateAttnMetadataPacked`，通过行轴区分前反向；
  未提供计数时，行轴为 0 选择 `CreateAttnMetadataForward`，行轴为 1 选择
  `CreateAttnMetadata`。算子名称不同不代表前向不支持 packed 模式。
- 输出是内部调度协议，不应手工修改。对齐产生的尾部空间不代表有效任务记录。

## 安装方式

在配置了 CANN、PyTorch 和 torch_npu 的 Linux 环境中，从仓库根目录执行：

```bash
bash build.sh --socs atlas_950 --install -j16
```

构建清单通过 `cmake/soc_ops_config.cmake` 登记 `atten_metadata`。目录内 CMake
分别构建 AI CPU 内核 `libtransformer_aicpu_kernels.so` 和 Host API `libcust_opapi.so`，
并安装至对应 SOC 的 `vendors/atten_metadata`。AI CPU 构建需要 CANN 提供的
`aarch64-target-linux-gnu-g++` 交叉编译器及开发头文件、库。

完整构建与加载说明见[编译指南](../../../../docs/build_guide.md)。

## 调用示例

以下示例为固定长度、无掩码的 MHA 前向生成 metadata：

```python
import torch
import torch_npu
import ops_rec

torch.npu.set_device(0)
metadata = ops_rec.ascendc.attention.create_attn_metadata(
    None, None, None, None,
    2, 1024, 1024,
    8, 8, 128,
    0, -1, -1,
    "BSND", "BSND", "BSND",
    packed_row_axis=0,
    deterministic=True,
)
print(metadata.shape, metadata.dtype, metadata.device)
```

变长、packed 及调度覆盖检查可参考
[基础测试](../test/pytest/test_create_attn_metadata.py)和
[百组用例测试](../test/pytest/test_create_attn_metadata_100_cases.py)。在仓库根目录执行：

```bash
python -m pytest ascendc/attention/atten_metadata/test/pytest -v
```

## ACLNN 接口说明

非 Python 调用可使用 `aclnnCreateAttnMetadataGetWorkspaceSize` 和
`aclnnCreateAttnMetadata` 两阶段接口，声明见
[aclnn_create_attn_metadata.h](../op_host/op_api/aclnn_create_attn_metadata.h)。

与 Torch 接口相比，ACLNN 将两个 packed 计数张量放在四个序列张量之后，
并在 `packedMaskHeads` 与 `qBlockSize` 之间增加 `l2Size` 参数，单位为字节。
Torch 层会查询设备 L2 大小并传入该参数。ACLNN 调用方需要预先分配输出张量；
不能直接照搬 Torch 接口的参数顺序。
