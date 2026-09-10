# ops-rec 算子列表

ops-rec 是面向大规模推荐场景的高性能算子库，基于昇腾 CANN 平台深度定制，采用 AscendC + Triton 的双后端架构。

算子仓目录按后端分组：`ascendc/`、`triton/`，另外还有独立的实验性算子目录 `experimental/`。安装后可按以下方式调用各算子：

- AscendC 算子：`ops_rec.ascendc.<group>.<op_name>(...)` or `from ops_rec.ascendc.<group> import <op_name>`
- Triton 算子：`from ops_rec.triton.<group>.<file_name> import <op_name>`
- 实验性算子：按后端复用 AscendC / Triton 调用方式，仅将 `ops_rec` 前缀替换为 `ops_rec.experimental`

本文档是 ops-rec 算子的索引与 Python 调用接口说明（当前已合入 AscendC 算子，Triton 算子与实验性算子将在后续版本补充）。AscendC 后端算子还可以通过 C 的 aclnn 接口调用，使用方法详见算子目录下的 aclnn 接口文档。

## 基本概念

| 术语 | 说明 |
|------|------|
| **HSTU** | Hierarchical Sequential Transduction Unit，分层序列转导单元，用于推荐场景的注意力机制 |
| **Jagged Tensor** | 一种不规则张量，各行长度可以不同，常用于变长序列场景 |
| **GQA** | Grouped Query Attention，分组查询注意力机制（K/V 头数小于 Q 头数） |
| **Silu** | Sigmoid Linear Unit 激活函数，公式为 `x * sigmoid(x)` |
| **LayerNorm** | 层归一化操作，对隐藏层进行均值方差归一化 |
| **af 编码** | Arbitrary Function 掩码编码，用每 query 4 个 INT32 描述其合法 KV 区间，供自定义注意力掩码使用 |
| **块稀疏** | 将注意力矩阵按块粒度分类为 FULL / MASK / EMPTY，跳过无关块实现高效计算 |

## 硬件支持说明

| 构建变体 | 芯片型号 | 说明 |
| --- | --- | --- |
| `atlas_a2` | Ascend910B* | Atlas A2 训练/推理系列 |
| `atlas_a3` | Ascend910_93* | Atlas A3 训练/推理系列 |
| `atlas_950` | Ascend950* | Atlas 950系列 |

> 运行时 `import ops_rec` 会通过 `acl.get_soc_name()` 自动检测芯片型号，并按 `cmake/soc_ops_config.cmake` 中的白名单加载对应算子实现。不同 SOC 支持的算子集合不同，同一算子在 950 上可能提供 FP8 等额外能力。

## AscendC 算子列表

### activation（激活算子）

| 算子 | 接口签名 | 功能简介 | 硬件支持 |
| ---- | -------- | -------- | -------- |
| [ln_mul](../../ascendc/activation/ln_mul/docs/torch_ln_mul.md) | `ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)` | 对输入 x 做 LayerNorm 后与 u 相乘，融合归一化与特征缩放 | Atlas A2, Atlas A3, Atlas 950 |
| [in_linear_silu](../../ascendc/activation/in_linear_silu/in_linear_silu_forward/docs/torch_in_linear_silu.md) | `ops_rec.ascendc.activation.in_linear_silu(x, weight, bias, split_arg_list)` | Linear + Silu 融合后拆分为 User/Value/Query/Key 四个向量（HSTU 注意力前处理），反向内部调用 `in_linear_silu_backward` | Atlas A2, Atlas A3, Atlas 950 |
| [in_linear_silu_backward](../../ascendc/activation/in_linear_silu/in_linear_silu_backward/docs/torch_in_linear_silu_backward.md) | `ops_rec.ascendc.activation.in_linear_silu_backward(x, weight, bias, user_grad, value_grad, query_grad, key_grad, linear_output, attr_dict)` | in_linear_silu 反向传播，计算 x、weight、bias 的梯度 | Atlas A2, Atlas A3, Atlas 950 |

### attention（注意力算子）

| 算子 | 接口签名 | 功能简介 | 硬件支持 |
| ---- | -------- | -------- | -------- |
| [hstu_dense](../../ascendc/attention/hstu_v1/hstu_forward_v1/docs/torch_hstu_forward.md) | `ops_rec.ascendc.attention.hstu_dense(q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=0, silu_scale=0.0)` | HSTU 融合注意力前向（dense 格式），支持 GQA 与 dim 不等 | Atlas A2, Atlas A3, Atlas 950 |
| [hstu_jagged](../../ascendc/attention/hstu_v1/hstu_forward_v1/docs/torch_hstu_forward.md) | `ops_rec.ascendc.attention.hstu_jagged(q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=0, max_seq_len_k=0, silu_scale=0.0, seq_offset=None, seq_offset_k=None, ...)` | HSTU 前向（jagged 格式），支持 equal / delta 双 overload 自动分派 | Atlas A2, Atlas A3, Atlas 950 |
| [hstu_paged](../../ascendc/attention/hstu_v1/hstu_forward_v1/docs/torch_hstu_forward.md) | `ops_rec.ascendc.attention.hstu_paged(q, k, v, kv_cache=None, ...)` | HSTU 前向（paged 格式，kv_cache 分页），支持增量推理 | Atlas A2, Atlas A3, Atlas 950 |
| [hstu_dense_backward](../../ascendc/attention/hstu_v1/hstu_backward_v1/docs/torch_hstu_dense_backward.md) | `ops_rec.ascendc.attention.hstu_dense_backward(grad, q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=0, silu_scale=0.0)` | HSTU 反向（dense 格式），计算 q/k/v/attn_bias 梯度 | Atlas A2, Atlas A3, Atlas 950 |
| [hstu_jagged_backward](../../ascendc/attention/hstu_v1/hstu_backward_v1/docs/torch_hstu_jagged_backward.md) | `ops_rec.ascendc.attention.hstu_jagged_backward(grad, q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len_q=0, max_seq_len_k=0, silu_scale=0.0, seq_offset_q=None, seq_offset_k=None, ...)` | HSTU 反向（jagged 格式），支持 equal / delta 双 overload | Atlas A2, Atlas A3, Atlas 950 |
| [hstu_backward_v2](../../ascendc/attention/hstu_v2/docs/torch_hstu_backward_v2.md) | `ops_rec.ascendc.attention.hstu_backward_v2(grad, q, k, v, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k, rab=None, ...)` | HSTU V2 反向，统一 jagged 格式，支持窗口掩码（causal / no mask） | Atlas 950 |
| [disentangle_attention](../../ascendc/attention/disentangle_attention/docs/torch_disentangle_attention.md) | `ops_rec.ascendc.attention.disentangle_attention(query_layer, key_layer, value_layer, pos_key_layer, pos_query_layer, relative_pos, attn_mask, pos_attr_type="c2p\|p2c", score_scale=1.0)` | DeBERTa 解耦注意力，支持 c2p / p2c 与双向三种位置编码模式 | Atlas A2, Atlas A3, Atlas 950 |
| [create_q2k_block_sparse_from_func](../../ascendc/attention/block_sparse_func/docs/torch_create_q2k_block_sparse_from_func.md) | `ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(func_tensor, q_len, kv_len, q_block_size, kv_block_size, check_q_boundary, debug=False)` | Q2K（正向）块稀疏张量生成，输出 mask/full 块数与块索引表 | Atlas 950 |
| [create_k2q_block_sparse_from_func](../../ascendc/attention/block_sparse_func/docs/torch_create_k2q_block_sparse_from_func.md) | `ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(func_tensor, q_len, kv_len, q_block_size, kv_block_size, debug=False)` | K2Q（反向/transpose）块稀疏张量生成，内部含 kv-range 优化 | Atlas 950 |
| [compact_no_sync](../../ascendc/attention/block_sparse_func/docs/torch_compact_no_sync.md) | `ops_rec.ascendc.attention.compact_no_sync(mask_block_cnt, full_block_cnt, block_idx)` | 块索引无同步压缩，输出 CSR 风格 offset 与压缩索引缓冲 | Atlas 950 |
| [build_hstu_mask_arbitrary_func](../../ascendc/attention/arbitrary_func/docs/torch_build_hstu_mask_arbitrary_func.md) | `ops_rec.ascendc.attention.build_hstu_mask_arbitrary_func(offsets, ctx_lens, tgt_lens, target_group_size, total_seqlen)` | 构造 HSTU 任意函数掩码（af 编码），区分 ctx/history/target | Atlas 950 |
| [build_sla_mask_arbitrary_func](../../ascendc/attention/arbitrary_func/docs/torch_build_sla_mask_arbitrary_func.md) | `ops_rec.ascendc.attention.build_sla_mask_arbitrary_func(offsets, total_seqlen, sla_k1, sla_k2, contextual_seq_len, num_targets=None)` | 构造 SLA（半局部注意力）任意函数掩码（af 编码） | Atlas 950 |
| [fa_infer_with_arbitrary_func](../../ascendc/attention/fa_infer_with_arbitrary_func/fa_infer_with_arbitrary_func_forward/docs/torch_fa_infer_with_arbitrary_func.md) | `ops_rec.ascendc.attention.fa_infer_with_arbitrary_func(q, k, v, arbitrary_func, q2k_block_sparse)` | Flash Attention 推理：任意区间掩码 + 块稀疏跳过，高效注意力前向 | Atlas 950 |

### mixing（混合算子）

| 算子 | 接口签名 | 功能简介 | 硬件支持 |
| ---- | -------- | -------- | -------- |
| [token_mixing](../../ascendc/mixing/token_mixing/docs/torch_token_mixing.md) | `ops_rec.ascendc.mixing.token_mixing(x, gamma, beta, epsilon=1e-7)` | Token 混合归一化：将张量与其转置相加后做 LayerNorm | Atlas A2, Atlas A3, Atlas 950 |

### sequence（序列算子）

| 算子 | 接口签名 | 功能简介 | 硬件支持 |
| ---- | -------- | -------- | -------- |
| [reverse_sequence](../../ascendc/sequence/reverse_sequence/docs/torch_reverse_sequence.md) | `ops_rec.ascendc.sequence.reverse_sequence(input, seq_lengths)` | 按 `seq_lengths` 指定的长度对每个 batch 的有效行进行逆序（其余行保持原值） | Atlas A2, Atlas A3, Atlas 950 |

> **注意**：上表中 `接口签名` 列为算子接口概览，完整的参数列表、类型约束、默认值及返回值说明请进入各算子对应的 docs 页面查看。

## 算子目录结构

```text
ascendc/
├── activation/                                    # activation 算子组（激活算子）
│   ├── ln_mul/                                    # LayerNorm + Multiply 融合
│   └── in_linear_silu/                            # Linear + Silu 融合（HSTU 前处理）
│       ├── in_linear_silu_forward/                # 前向
│       └── in_linear_silu_backward/               # 反向
├── attention/                                     # attention 算子组（注意力算子）
│   ├── hstu_v1/                                   # HSTU V1（dense / jagged / paged 三种布局）
│   │   ├── hstu_forward_v1/                       # 前向
│   │   └── hstu_backward_v1/                      # 反向
│   ├── hstu_v2/                                   # HSTU V2 反向（统一 jagged 格式，仅 Atlas 950）
│   ├── disentangle_attention/                     # 解耦注意力 (DeBERTa)
│   ├── block_sparse_func/                         # 块稀疏张量生成（FULL/MASK/EMPTY 分类）
│   ├── arbitrary_func/                            # 任意函数掩码 af 编码（HSTU / SLA）
│   └── fa_infer_with_arbitrary_func/              # Flash Attention 推理（任意区间掩码）
├── mixing/                                        # mixing 算子组（混合算子）
│   └── token_mixing/                              # Token 混合归一化
└── sequence/                                      # sequence 算子组（序列算子）
    └── reverse_sequence/                          # 序列逆序
```

## 算子实现说明

- **AscendC 算子**：`ascendc/` 目录下每个**叶子算子**都包含 `op_host/`（tiling 与算子注册）、`op_kernel/`（AscendC kernel）、`docs/`（aclnn + torch 双接口文档）、`examples/`（aclnn + torch 示例）与 `test/`。`torch_extension/`（torch 适配层，注册 `torch.ops.ops_rec.*`）默认位于叶子算子根目录；对于 `in_linear_silu`、`hstu_v1`、`fa_infer_with_arbitrary_func` 这类聚合算子，`torch_extension/` 上提到聚合根目录，由其下的 forward/backward 等子算子共用。
- **SOC 白名单**：各 SOC 编译哪些算子由 `cmake/soc_ops_config.cmake` 的 `OPS_REC_OPS_ATLAS_950 / _A3 / _A2` 三个变量控制。构建时未列入白名单的算子在对应芯片上不可用。
- **统一 Python 调用接口**：`import ops_rec` 自动按本机 SOC 加载 `ops_rec_atlas_<variant>.so`（torch 适配层），AscendC 算子按 `ops_rec.ascendc.<group>.<op_name>(...)` 形式调用；同时提供 C 语言的 aclnn 接口供非 Python 场景使用，详见各算子目录下的 `aclnn` 接口文档。
- **Triton / 实验性算子**：`triton/` 与 `experimental/` 源码目录当前为空，将在本索引后续版本补充对应章节。
