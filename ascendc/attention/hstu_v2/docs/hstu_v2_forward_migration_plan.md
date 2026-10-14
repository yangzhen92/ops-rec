# hstu_v2 前向迁移方案（只迁前向 · 反向完全不动）

> 方案日期：2026-09-22
> **基线 HEAD**：`24b3dd05ed67442ac1732e931d3be88125ca5eea`（简称 **B**）
> **验收标准**：与 B 相比，**反向完全不动** ∧ **能正常运行** ∧ **功能完备**
> 证据基础：[hstu_v2_forward_migration_analysis.md](hstu_v2_forward_migration_analysis.md)（三仓对照分析）

---

## 0. 范围与验收标准

**做**：以 `24b3dd05` 为基线，把 lihao 分支（`optim_block_sparse_new` @ `895f90b`）的**前向能力**搬进本仓，使前向达到与反向对等的功能面（含 packed / arbitrary mask）。

**不做**：
- 不修反向的任何既有问题、错漏、性能缺陷
- 不把反向升级到上游的 L1 复用（`L1BlockReuseCache` / `Probe`）设计
- 不与上游对齐而改反向的任何一行

### 0.1 标准的三条可核查判据

| # | 判据 | 核查命令 | 通过条件 |
|---|---|---|---|
| **S1** | 反向**专属文件**逐字节不动 | `git diff 24b3dd05 HEAD -- <§2-B 类文件清单>` | **完全空** |
| **S2** | 反向**共享文件**只允许白名单内纯新增 | `git diff 24b3dd05 HEAD -- <§2-A 类共享文件>` | 无任何删除行；新增行全部落在 §3 白名单内 |
| **S3** | 反向**能跑且功能完备** | 反向精度/性能 case 与 `24b3dd05` 构建的 `.so` 对比 | 逐 case 数值一致 |

**S2 的实质**：白名单里的每一处新增都必须在「反向路径上恒不激活」，即**反向的控制流与 B 逐位相同** —— 论证见 §3，不是"等价性推导"，而是"不进入"。

---

## 1. 核心结论

> **本方案不需要改写任何既有函数的语义。** 前向所需的全部能力都能以「新文件 + 白名单纯新增」的形式加在 B 之上。

原方案（取 D 版再改回）里那处 `GetTriggerSwizzle()` 契约冲突（读后清零 → 粘滞只读）**在新标准下不存在** —— 因为前向对 swizzle API 的引用数为 0：

```
kernel/fwd/*.hpp + op_kernel/hstu_forward_v2*.cpp 内
  grep GetTriggerSwizzle | GetSwizzlePosition | L1BlockReuseCache | .Probe(   →  0 命中
```

那整套状态机（粘滞 getter + `swizzlePosition` + `L1BlockReuseCache`）是 D 为**反向 L1 复用**引入的，前向一行都不读。**保持 B 版即可，不需要动。**

### 1.1 前向对调度器的完整调用面（穷举，14 个方法）

`EnablePackedSparse`、`GetBlockId`、`GetSeqLens`、`GetTile`、`GetTileMapping`、`Init`、`IsFirst`、`IsLast`、`IsValid`、`GetCurrentSeqLen`、`GetHeadBlockId`、`GetMeta`。

其中只有 `EnablePackedSparse` 是 B 版没有的 → 这是 `block_scheduler.hpp` 唯一必须新增的能力。

### 1.2 前向的调度器拓扑（两条路径，运行期二选一）

`op_kernel/hstu_forward_v2.cpp:85 / :104` 按 `metadata != nullptr` 运行期择一：

| 分支 | Q（行）调度器 | K（列）调度器 | 触发条件 |
|---|---|---|---|
| **metadata 驱动** | `MetadataRowBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>` | `ColumnBlockScheduler<MetaQ…, BLOCK_M, BLOCK_N, true>` | `metadata != nullptr`（B=1 的 AIV schedule 路径） |
| **设备现算（默认）** | **`InterleavedRowBlockScheduler<…>`** | `ColumnBlockScheduler<Interleaved…, BLOCK_M, BLOCK_N, false, false, HSTU_FWD_KV_CLUSTER_NUM>` | `metadata == nullptr`（多 batch / 非 packed 路径） |

第二条路径的 Q 调度器由宏切换（`kernel/fwd/forward_kernel_builder.hpp:36`，**该 `#define` 处于生效状态**）：

```cpp
#define USE_INTERLEAVED_Q_SCHEDULER      // ← 生效：交错分核，提升 L2 跨核共享率
```

> ⚠ 第 33-34 行只是操作说明注释，**第 36 行的 `#define` 才是实际状态** —— 前向默认走 `InterleavedRowBlockScheduler`。

两条路径的 K 调度器都是 `ColumnBlockScheduler`，都住在 `block_scheduler.hpp` 里 —— 这是 §3.1 那个白名单文件的由来（packed 跳过逻辑在 `ColumnBlockScheduler` 上，两条路径共用）。

`MetadataRowBlockScheduler` 是**独立类**（`class MetadataRowBlockScheduler {`，无基类），与 `RowBlockScheduler` 的既有改动完全解耦 → §3.1 里 `RowBlockScheduler` 的一切 D 版改动都可以不取。

---

## 2. 文件清单（按「反向是否依赖」三分）

分类由 `migrate_guard.py` 闭包扫描产出：前向闭包 31 个、反向闭包 36 个、交集 13 个。

### A 类：共享文件（前向 ∩ 反向，13 个）

| 文件 | B→D | **本方案取值** | 反向风险 |
|---|---|---|---|
| `common/hstu_metadata_layout.h` | 新增 | D 版（新文件） | 无（B 反向不可能引用新文件） |
| `op_kernel/catlass_hstu/detail/target_block_range.hpp` | 新增 | D 版（新文件） | 无 |
| `op_kernel/catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp` | 新增 | D 版（新文件） | 无 |
| `op_kernel/catlass_hstu/kernel/mask/arbitrary_mask_predictor.hpp` | 新增 | D 版（新文件） | 无 |
| `op_kernel/catlass_hstu/kernel/pipeline_policy.hpp` | 新增 | D 版（新文件） | 无 |
| **`op_kernel/catlass_hstu/gemm/block/block_scheduler.hpp`** | +324/-11 | **B 版 + packed 纯增量** | ★ §3.1 |
| **`op_kernel/catlass_hstu/kernel/mask/no_mask_predictor.hpp`** | +20/-1 | **B 版 + 3 个纯新增方法** | ★ §3.2 |
| `op_kernel/catlass_hstu/gemm/dispatch_policy.hpp` | +1/-2 | **保持 B 版** | 无（纯格式，逐字一致） |
| `op_kernel/catlass_hstu/gemm/tile/copy_gm_to_l1_a5.hpp` | +3/-3 | **保持 B 版** | 无（纯格式） |
| `op_kernel/tla_hstu/layout.hpp` | +3/-3 | **保持 B 版** | 无（纯格式） |
| `torch_extension/hstu_v2_param_check.h` | +42/-0 | **D 版：纯追加 42 行**（§3.3，选项 a） | 无（零删除；反向插件零引用新类） |

> 另 2 个共享文件三方逐字节一致，本方案不动：`gemm/block/block_scheduler_interleaved.hpp`（前向默认 Q 调度器）、`gemm/tile/copy_l0c_to_ub_a5.hpp` 等。
>
> 三个「纯格式」文件已逐字核对：`dispatch_policy.hpp` 只差模板参数列表折行；`copy_gm_to_l1_a5.hpp` 只差 `{};`→`{}`、`const&`→`const &`、命名空间注释；`tla_hstu/layout.hpp` 只差注释缩进。**B 版对前向已够用，改它们只会徒增反向 diff。**

### B 类：反向专属文件 —— 迁移时必须整体回退到 `24b3dd05`

**S1 判据的文件清单**（`git diff 24b3dd05 HEAD -- <本清单>` 必须为空）：

```
op_host/hstu_backward_v2.cpp                                   (+115/-2)
op_host/hstu_backward_v2_define.h                              (+12/-1)
op_host/hstu_backward_v2_tiling.h                              (+4/-0)
op_kernel/hstu_backward_v2.cpp                                 (+70/-28)
op_kernel/hstu_backward_v2_packed.cpp                          (新增 → 不迁)
op_kernel/catlass_hstu/kernel/bwd/backward_kernel_builder.hpp  (+4/-27)
op_kernel/catlass_hstu/kernel/bwd/backward_kernel_epilogue_mainloop.hpp  (+47/-17)
op_kernel/catlass_hstu/kernel/bwd/backward_kernel_mmad_mainloop.hpp      (+84/-21)
op_kernel/catlass_hstu/kernel/bwd/backward_kernel_resource.hpp           (+9/-1)
op_kernel/catlass_hstu/gemm/block/block_mmad_dq.hpp            (+4/-4)
op_kernel/catlass_hstu/gemm/block/block_mmad_pv.hpp            (+9/-3)
op_kernel/catlass_hstu/gemm/block/block_mmad_qk.hpp            (+11/-4)
op_kernel/catlass_hstu/gemm/block/l1_block_reuse_cache.hpp     (新增 → 不迁)
op_kernel/catlass_hstu/epilogue/block/block_epilogue_rab_grad.hpp      (+9/-9)
op_kernel/catlass_hstu/epilogue/block/block_epilogue_score_grad.hpp    (+61/-2)
op_kernel/catlass_hstu/epilogue/regbase/fast_rab_grad.hpp              (+3/-3)
op_kernel/catlass_hstu/epilogue/regbase/fast_silu_grad.hpp             (+105/-0)
op_kernel/catlass_hstu/kernel/mask/causal_mask_predictor.hpp           (+69/-11)
op_kernel/catlass_hstu/kernel/mask/predictor_builder.hpp               (+32/-11)
torch_extension/hstu_backward_v2_plugin.cpp                            (+49/-34)
```

> **`block_mmad_{dq,pv,qk}.hpp` 在 B 类而非 A 类**：前向用的是 `block_mmad_qk_infer.hpp` / `block_mmad_pv_infer.hpp`（三方逐字节一致），与反向**不共文件**。`l1_block_reuse_cache.hpp` 是反向 L1 复用所需，本方案**不迁**。
>
> `causal_mask_predictor.hpp` / `predictor_builder.hpp` 既是反向专属、又**不能取 D 版**：D 版把 predictor 状态从 `kernel->xxx` 改成 `this->xxx` + `Construct()` 注入，而 B 的反向 mainloop 没有 `Construct()` 调用点 → 编译能过但**运行期解空指针 / 整数除零**。必须整取 B 版。

### C 类：前向专属的既有文件（反向零引用，取 D 版）

```
op_kernel/catlass_hstu/epilogue/block/block_epilogue_score_infer.hpp   (+29/-6)  ← 反向用 score_grad
op_kernel/catlass_hstu/epilogue/regbase/fast_silu_infer.hpp            (+54/-0)  ← 反向用 fast_silu_grad
```

已 grep 确认反向 B 版全树零引用（不在反向闭包内）。

### D 类：纯新增（反向零引用，零风险）

```
op_kernel/catlass_hstu/kernel/fwd/*.hpp                     (builder / mmad_mainloop / epilogue_mainloop / resource / tiling)
op_kernel/hstu_forward_v2.cpp / hstu_forward_v2_packed.cpp
op_host/hstu_forward_v2.cpp / _define.h / _tiling.h
torch_extension/hstu_forward_v2_plugin.cpp / hstu_common.h
torch_extension/hstu_forward_v2_param_check.h               (见 §3.3)
test/pytest/test_forward_op_*.py、test_pipeline_combinations.py
```

> `common/hstu_schedule_policy.h`（`hstu::schedule::WeightedTarget/UniformBoundary`）**全仓零引用**，是悬空占位文件 —— **不迁**。
>
> **构建文件无需改动**：`op_kernel/CMakeLists.txt` 与 `op_host/CMakeLists.txt` 都是 GLOB 收集，`cmake/soc_ops_config.cmake` 里 hstu_v2 已在列。

---

## 3. 共享文件白名单（S2 的全部内容）

### 3.1 `block_scheduler.hpp`

**要加的（全部纯新增）**

| 新增项 | 位置 | 谁用 |
|---|---|---|
| `usePackedSparse{false}` / `sparseFirst{true}` 成员 | `ColumnBlockScheduler` private | — |
| `EnablePackedSparse(...)` | `ColumnBlockScheduler` public | 前向 `forward_kernel_{mmad,epilogue}_mainloop.hpp:291/272` |
| `IsPackedNonEmpty` / `FindNextNonEmpty` | `ColumnBlockScheduler` | 被下面 4 个分支调用 |
| 4 个既有函数内插 `if (usePackedSparse) {...}` 分支 | `Init`（`Update()` 前）、`IsFirst`、`IsLast`、`IsValid` | packed 跳过语义 |

**不加的（D 有、前向不读 → 保持 B 版）**

```
# 反向 L1 复用（D 为反向引入，前向零引用）
GetTriggerSwizzle() 的语义改写   ← 保持 B 的「读后清零」，一字不改 ★
GetSwizzlePosition() / swizzlePosition 成员 / Init 里的复位行
operator++ / SkipBlocks 里的 swizzlePosition 递增
l1_block_reuse_cache.hpp 整文件
# workload 分核（调用点全在 D 的反向文件，B 的反向不调）
EnableTargetWorkload / EnablePackedSparseWorkload / TargetWorkloadSplitCore
GetKBlockWorkload / GetTargetQBlockEnd / GetSparseRowWorkload / PackedSparseSplitCore
useTargetWorkload / usePackedSparseWorkload 成员
# RowBlockScheduler 的既有改写
IsLast() 去括号、SplitCore 的 if/else-if 三分支、被删的注释行
```

**为什么对反向中性（S2 论证）**

1. 反向路径上 `usePackedSparse` **恒为 `false`** —— 唯一置 `true` 的入口是 `EnablePackedSparse`，其调用点在前向闭包内（`kernel/fwd/` 2 处），反向闭包零调用。
2. 因此那 4 个插入分支**全部不进入**，控制流与 B 逐位相同。
3. `GetTriggerSwizzle()` 保持 B 的读后清零 → 反向 mainloop 的「跳过 GM→L1 载入」开关语义原样保留。
4. **剥离是机械干净的**：D 版里 `swizzlePosition` 只出现在 `634/684/767/773/781/926`，`usePackedSparse`/`sparseFirst` 只出现在 `612/654/655/696/697/711/727/761/765/919/920` —— **零重叠行**，两块功能在源码上不交织。

### 3.2 `no_mask_predictor.hpp`

**要加的（3 个纯新增方法，`NoMaskPredictor` 内）**

| 新增方法 | 谁调 |
|---|---|
| `Construct(Kernel*, typename Kernel::Params const&)`（空实现） | `forward_kernel_epilogue_mainloop.hpp:282` —— **无条件调用** |
| `UsesDirectScoreMask() const { return false; }` | `block_epilogue_score_infer.hpp:120` —— **在 `if constexpr` 之外**无条件调用 |
| `CopyPackedMask(...)`（空实现） | `block_epilogue_score_infer.hpp:116`（在 `if constexpr (HAS_MASK)` 内，为接口齐整一并加） |

**不改的**：`ApplyMask` **保持 B 的 `void`**。前向对 `ApplyMask` 零引用；D 的 `bool` + `return false` 是给反向 `block_epilogue_score_grad.hpp:130` 的 `needMask = predictor.ApplyMask(...)` 用的，**不迁**。

**为什么对反向中性（S2 论证）**

反向对 `NoMaskPredictor` 的调用面只有 `MakeBlockPredParams` / `Classifier` / `IsSkip` / `ApplyMask`：前者三方逐字节相同，`ApplyMask` 保持 B 签名 → 反向源文件与 B 版完全一致；新增的 3 个方法反向零调用。

### 3.3 `torch_extension/hstu_v2_param_check.h`（**已定：选项 a**）

D 版在此文件追加了 42 行新类 `HstuForwardV2ParamChecker`（**零删除**），该文件同时被前向和后向插件 include。

| 选项 | 做法 | 后果 |
|---|---|---|
| **a ✅ 已选定** | 直接追加 D 的 42 行 | 反向插件行为不受影响（纯增）；该文件 diff 非空，**列进 §3 白名单**，S2 判据覆盖 |
| b | 塞进新头文件 `torch_extension/hstu_forward_v2_param_check.h`，前向插件改 include | 该共享文件保持 B 版逐字节；本次不采用 |

**白名单登记（S2 判据）**：`torch_extension/hstu_v2_param_check.h` 相对 B 只允许出现新增行，且新增行只能是 `HstuForwardV2ParamChecker` 类（+42 行）；**不允许任何删除行**。反向插件对该类的引用数为 0。

---

## 4. 执行步骤

**起点**：`24b3dd05`（B）。若从当前 `b8b00209` 出发，先 `git revert b8b00209` 或直接以 B 开新分支重放。

| 阶段 | 内容 | 涉及 |
|---|---|---|
| **1** | 铺共享层新文件（4 个新增 + `hstu_metadata_layout.h` + `target_block_range.hpp`） | 纯新增 |
| **2** | `block_scheduler.hpp` 最小加性改动（§3.1 白名单） | ⚠ 唯一有技术风险的一步 |
| **3** | `no_mask_predictor.hpp` 加 3 个方法（§3.2） | 纯新增 |
| **4** | 前向主体（kernel/fwd、op_kernel/op_host/torch_extension/test） | 纯新增 |
| **5** | 前向专属既有文件取 D 版（`block_epilogue_score_infer.hpp` / `fast_silu_infer.hpp`） | 反向零引用 |
| **6** | 编译 + 跑 §5 验证 | — |

阶段 1/3/4/5 均为纯新增或反向零引用，阶段 2 是**唯一**需要 `block_scheduler.hpp` 常规审查的一步。

---

## 5. 验证方案

| # | 验证项 | 方法 | 判据 |
|---|---|---|---|
| **V1** | **S1：反向专属文件零改动** | `git diff 24b3dd05 HEAD -- <§2-B 类清单>` | **空 diff** |
| **V2** | **S2：共享文件只有白名单纯新增** | `git diff 24b3dd05 HEAD -- <§2-A 类中保留 B 版的文件 + block_scheduler.hpp + no_mask_predictor.hpp>` | 逐 hunk 检查：**无删除行**；新增行全部落在 §3 白名单 |
| **V3** | 反向编译通过 | 全量编译，反向 kernel + host + plugin 均通过 | 无 error |
| **V4** | 反向 ABI 未变 | 反向 kernel 入口 `GM_ADDR` 个数、`ASCENDC_TPL_SEL_PARAM` 参数个数与 `24b3dd05` 对比 | 逐字一致 |
| **V5** | **S3：反向精度未回归** | `24b3dd05` 与迁移后各构建一版 `.so`，跑同一组 `test_backward_op_precision*.py`（固定 seed） | 逐 case 数值一致 |
| **V6** | **S3：反向性能未回归** | `run_benchmark.sh` backward 段 + `perf_stats.py` | Task Duration 与基线同量级 |
| V7 | 前向精度 | `test_forward_op_precision*.py` 全量 | 达标 |
| V8 | 前向 packed / arbitrary | `test_forward_op_precision_arbitrary_mask.py`，**必须同时覆盖 §1.2 的两条调度路径**：`batch_size=1` 走 metadata 路径，`batch_size=8` 走 Interleaved 路径 | 两者都过 |
| V9 | 前后向一致性 | `test_pipeline_combinations.py` 四种 pipeline 组合 | 四种路径 `allclose` |

**V5/V6 的性质变了**：反向源码与 B 完全一致（除白名单），所以反向的**正确性论证不需要重做** —— V5/V6 的作用是**证明白名单改动确实中性**（若结果与 `24b3dd05` 构建的 `.so` 逐 case 一致，说明 §3 的"恒不激活"论证成立）。

**V2 是关键**：它是新标准下唯一需要人工逐 hunk 审查的判据。判据不是"结果等价"，而是"**没有删除行**"。

---

## 6. 与已落地 HEAD（`b8b00209`）的关系

| | `b8b00209`（`origin/hstu_v2_fwd`，已推送） | 本方案 |
|---|---|---|
| 前向 | 完整落地 | 完整落地（功能面一致） |
| 反向源码 | 成套升级（L1 复用 + packed 反向 + arbitrary mask 真实现） | **零改动**（= B 版） |
| 反向 arbitrary mask | 可用（真实现） | 不可用（B 原行为：静默当 dense） |
| 反向 L1 复用 | 有 | 无（= B 版，无此功能） |
| 与 lihao `block_scheduler.hpp` 分叉 | 0 | ~150 行（swizzle + reuse + workload 三块不取） |
| 回滚 | 需 revert 整个提交 | 阶段 2 单文件回滚 |

**若采纳本方案**：`b8b00209` 需要回退或重做 —— 建议以 `24b3dd05` 为起点按 §4 重放。

> **必须在动手前确认的一个前提**：本方案下反向**保持 B 原行为**，包括「传 arbitrary mask 会被静默当成 dense 处理」这一点。这是迁移前就存在的既有行为，本方案是「维持原状」而非「引入缺陷」，但**需要向使用方明确**。

---

## 7. 回滚

- 阶段 1/3/4/5 均为纯新增或反向零引用文件，`git revert` 即可
- 阶段 2 是唯一改动共享文件的一步 → 恢复 B 版 `block_scheduler.hpp` 即可完全回滚

---

## 8. 本方案明确不做的事

| # | 不做 | 理由 |
|---|---|---|
| 1 | 不把反向升级到 L1 复用（`L1BlockReuseCache` / `Probe` / mmad 的 `l1ReuseSlot`） | 超出「只迁前向」范围；且属性能增强，需重做反向精度验证 |
| 2 | 不修反向 `isArbitrary` 只看 `arbitraryFunc` 的问题 | 反向既有问题，不迁 |
| 3 | 不给反向补 `LegacyTilingFunc` 保护 | 同上 |
| 4 | 不恢复被删的 2 个反向 `test_generic_graph` 用例 | 属 `b8b00209` 的产物；以 `24b3dd05` 为起点重放时本就不存在 |
| 5 | 不迁 `common/hstu_schedule_policy.h` | 全仓零引用的悬空占位 |
| 6 | 不动 `dispatch_policy.hpp` / `copy_gm_to_l1_a5.hpp` / `tla_hstu/layout.hpp` | 与 B 仅差 clang-format，B 版对前向已够用 |
| 7 | 不改 `predictor_builder.hpp` / `causal_mask_predictor.hpp` | 反向专用；前向零引用 |
| 8 | 不改 `no_mask_predictor.hpp` 的 `ApplyMask` 签名 | 前向零引用；改了只会给反向引入 diff |
| 9 | 不改 `GetTriggerSwizzle()` / `RowBlockScheduler` | 前向零引用；改了会破坏 B 反向的 L1 载入开关语义 |

---

## 附录：一句话版本

> 以 `24b3dd05` 为基线，前向所需的一切能力都能以「**4 个新文件 + 1 个新头文件 + 2 处白名单纯新增**」的形式加进去：`block_scheduler.hpp` 只加 packed-sparse 支持（`usePackedSparse` 默认 `false`，反向零激活），`no_mask_predictor.hpp` 只加 3 个前向调用的空方法。**反向的每一个文件都与 `24b3dd05` 逐字节相同，`GetTriggerSwizzle()` 一字不改** —— 反向不是"等价"，是"没动"。
