# Attn AI CPU 接口与 metadata 布局

本文对应从 ops-rec_lihao 迁入的实现（2026-10-19）。接口以实际源码为准。

## 1. 构建与调用链

Torch → ACLNN → L0 → AI CPU 分核/序列化 → HSTU consumer。
算子接收长度、调度属性及可选 packed 计数，不接收 Q/K/V 数据。
atten_metadata 已加入 atlas_950 白名单；使用仓库 build.sh 构建 wheel，运行时先 import ops_rec。
构建系统自动收集本目录 torch_extension 并安装 vendor。

- [Torch 注册和分配](torch_extension/create_attn_metadata_plugin.cpp)
- [ACLNN 声明](op_host/op_api/aclnn_create_attn_metadata.h)
- [L0 路由](op_host/op_api/l0_create_attn_metadata.cpp)
- [AI CPU 实现](op_kernel_aicpu/create_attn_metadata_aicpu.cpp)
- [共享 ABI](../attn_plan/common/attn_metadata_layout.h)

## 2. Torch 接口

```python
metadata = torch.ops.ops_rec.create_attn_metadata(
    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
    batch_size, max_seqlen_q, max_seqlen_kv,
    num_heads_q, num_heads_kv, head_dim,
    mask_mode, win_left, win_right,
    layout_q, layout_kv, layout_out,
    mask_block_cnt=None, full_block_cnt=None,
    packed_row_axis=0, packed_mask_heads=1,
    q_block_size=0, kv_block_size=0, deterministic=True,
)
```

也可经 ops_rec.ascendc.attention.create_attn_metadata 动态代理调用。
当前没有 is_grad_enabled 参数、WithGrad 入口或 FAG 尾部；HSTU backward 通过 packed_row_axis=1 指定。

| 参数 | 约束/语义 |
| --- | --- |
| cu_seqlens_q/kv | 可选 NPU INT32 [B+1]，从 0 开始的非递减累计边界 |
| seqused_q/kv | 可选 NPU INT32 [B]，各 batch 独立实际长度 |
| batch_size | 正数或 -1；Torch 在 -1 时从 Q 侧 seqused 或 cu 推导 |
| max_seqlen_q/kv | 正数或 -1；缺少对应长度 tensor 时必须提供长度 |
| num_heads_q/kv | AI CPU 要求正数且相等，仅支持 MHA |
| head_dim | 16 的正整数倍且不超过 256；HSTU 测试取 max(dimQK,dimV) |
| mask_mode | 0 无掩码，3 右下对齐 causal，4 band |
| win_left/right | -1 表示无限；mask 0/3 要求两者均为 -1 |
| layout_q/out | BSND、BNSD、TND |
| layout_kv | 另支持 PA_BNBD、PA_BBND、PA_NZ；PA 要求 seqused_kv |
| packed_row_axis | 0 为 Q 外轴，1 为 K 外轴；无 packed 时也控制方向 |
| q_block_size/kv_block_size | 同时为 0 使用默认值，否则同时为正；按原始 Q/KV 语义传入 |
| deterministic | True 完整行调度；False 允许 Stream-K |

TND 必须提供对应 cu_seqlens；非 TND 不允许对应 cu_seqlens。
实际长度优先取 seqused，其次 cu_seqlens，最后 max_seqlen。
Torch 对提供的 tensor 调用 contiguous；ACLNN 调用方自行保证连续性。

packed 两个计数成对提供，均为 NPU INT32 [B,Hmask,O]，形状相同。
Hmask=packed_mask_heads，取 1 或 num_heads_q；1 表示共享计数。
连续索引为 (b*Hmask+h)*O+r，字节偏移为索引乘 4。
行权重为 mask_count+full_count，O 必须覆盖有效外轴块数：
axis=0 为 ceil(Lq/Qblock)，axis=1 为 ceil(Lkv/KVblock)。
实际 sparse 索引等仍通过 HSTU 的九个 sparse_info tensor 传递。

## 3. ACLNN 接口

ACLNN 的计数参数位于长度 tensor 之后，位置与 Torch 不同；l2Size 单位为 bytes，Torch 自动查询。

```cpp
aclnnStatus aclnnCreateAttnMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKvOptional,
    const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    const aclTensor* maskBlockCntOptional, const aclTensor* fullBlockCntOptional,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t maskMode, int64_t winLeft, int64_t winRight,
    const char* layoutQ, const char* layoutKv, const char* layoutOut,
    int64_t packedRowAxis, int64_t packedMaskHeads, int64_t l2Size,
    int64_t qBlockSize, int64_t kvBlockSize, bool deterministic,
    const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnCreateAttnMetadata(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

输出 metaData 由调用方预分配。当前实现将 workspaceSize 写为 0；调用方仍应按返回值处理。
输入、输出及 workspace 保活到 stream 工作完成，consumer 建立正确的 stream 依赖。
ACLNN 在调用 L0 前从原始 Q 侧长度推导 batch_size=-1；无法推导时立即报错。
AI CPU 再次校验所有长度 tensor 的 batch、类型、非负性及累计边界，不允许负 batch 转为无符号循环次数。
Torch 查询当前线程资源，而当前 ACLNN 使用 PlatformInfo 核数；两侧尚未统一受限资源，
使用时应确保核数一致，尤其不能按受限核数分配更小的输出。

## 4. 方向与 tile

| 条件 | AI CPU op type | 外轴/内轴 |
| --- | --- | --- |
| 无 packed，axis=0 | CreateAttnMetadataForward | Q/KV |
| 无 packed，axis=1 | CreateAttnMetadata | K/Q |
| 有 packed | CreateAttnMetadataPacked | 由 axis 决定 |

普通 backward 的 L0 交换长度和最大长度；packed backward 保留原始输入语义。
显式 tile：forward 的 Mbase/S2base 为 Qblock/KVblock，backward 为 KVblock/Qblock。
默认 forward：head_dim<=128 时 128/128，否则 64/64；
默认 backward：head_dim<=128 时 128/256，否则 64/128。
HSTU 应显式传入 consumer 实际选择的 tile，避免默认值和动态 tile 不一致。

## 5. 输出 ABI

输出为连续一维 NPU INT32 tensor，内部按 uint32 word 写入。
1 word=4 bytes，record=16 words=64 bytes。S 为实际 section 数，C/V 为 AIC/AIV 核数。

| HEAD word | 内容 |
| --- | --- |
| 0 | S |
| 1 | isFd；当前 HSTU 关闭 FD |
| 2/3 | Mbase/S2base |
| 4..12 | 保留、清零；没有有效核数、FAG offset 或 split mode 字段 |
| 13/14 | V/C |
| 15 | magic：0 坐标，0x48535455 direct，0x534B4855 Stream-K |

普通布局：HEAD[16]+FA[S][C][16]+FD[S][V][16]。
FA 地址为 16+(s*C+c)*16；FD 起点为 16+S*C*16，均以 word 为单位。
全部 FA section 连续排列后才是 FD；无 FD 任务时 FD 内容未定义。

| 模式 | FA record 内容 |
| --- | --- |
| 坐标 | word 0..6：bn2Start,mStart,s2Start,bn2End,mEnd,s2End,firstFdDataWorkspaceIdx；其余为 0 |
| packed/direct | word 0/1：全局行块起点/行块数；其余为 0，S=1 |
| Stream-K | word 0/1：全局二维 tile 起点/tile 数；其余为 0 |

bn2=batch*H+head；m/s2 是块坐标，坐标描述线性遍历边界。
行块顺序为 batch→head→外轴行块，二维 tile 再追加内轴列块。
区间为半开区间，空闲核可有空 record，不能假设最后一个核必有工作。
packed/direct 按计数加权，以 NEAREST 策略选择完整行边界。
consumer 先看 magic，并使用 HEAD[14] 的 C 计算 stride。

普通/Stream-K 最小容量为 16+S*(C+V)*16 words；确定性 packed/direct 为 16+C*16 words。
Torch 使用 Smax=max(1,B*num_heads_kv)，普通保守容量为 16+Smax*(C+V)*16 words，
最终向上对齐到 4096 words（16 KiB）。padding 不保证清零，不能从 numel() 推导有效 section 数。

## 6. HSTU pytest meta 分支审查

utils/feature_matrix.py 的 METADATA_ARBITRARY 分支按方向调用 plan 生成 sparse_info 和 tile，
再显式调用 create_forward_metadata/create_backward_metadata；plan 自带 metadata 被丢弃。
因此它测试独立 AI CPU metadata 与 HSTU 的组合，区别于 PLAN 分支。

- sparse_info[0]/[3] 为 mask/full 计数，shape[1] 为共享或逐 head 数。
- forward 使用 axis=0，backward 使用 axis=1；tile 按 Qblock/KVblock 传递。
- TND offset 为 INT32；mask_mode=0、win=-1 正确，任意 mask 工作量由 packed 计数提供。
- 默认 deterministic=True，覆盖 packed/direct，未覆盖非确定性 Stream-K。
- 上述为源码契约结论；精度和运行时兼容性需在 Ascend/CANN/torch_npu 环境执行 pytest。

## 7. PR #180 检视修复与回归

已处理 [PR #180 检视意见](https://gitcode.com/Ascend/ops-rec/pull/180/diffs)：

- mask 0/3 拒绝非 -1 窗口；plan 将规则掩码的窗口规范化为 -1，band 保留原窗口。
- 普通 backward 在 L0 同时交换 Q/KV 输入、长度和 layout，layout_out 保持不变。
- Torch、ACLNN 和 AI CPU 统一要求 MHA；batch=-1 必须有可推导的 Q 侧长度。
- 普通/Stream-K 在写入前校验完整输出容量；metadata 字段访问失败返回状态，不触发 assert/abort。
- packed 在遍历前校验两个计数 tensor 的 dtype、形状、存储大小和行容量，并检查非负计数及行权重溢出。
- packed 的非 TND seqused 按独立长度读取；仅累计格式相减，避免长度递减时无符号下溢。

`test/pytest/test_metadata_review_regressions.py` 覆盖混合 layout、非对称 tile、独立 seqused 及 Host 参数拒绝。
100-case 数据位于同目录 `metadata_cases.py`，无需依赖仓库外 benchmark；packed 计数和 metadata 显式使用同一 tile。

在安装匹配源码的 atlas_950 wheel、CANN 和 torch_npu 后，从仓库根目录运行：

```bash
python -m pytest ascendc/attention/atten_metadata/test/pytest -q
cd ascendc/attention/hstu_v2/test/pytest
python -m pytest test_forward_precision.py test_backward_precision.py -q
python -m pytest test_msanitizer.py -k metadata_arbitrary -q
```

最后一条为普通执行；内存及同步插桩检查需另由 mssanitizer 启动。
