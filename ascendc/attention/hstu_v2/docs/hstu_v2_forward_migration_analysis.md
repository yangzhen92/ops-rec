# hstu_v2 前向迁移：三仓对照分析与「不影响反向」迁移方案

> 分析日期：2026-09-22
> 分析对象：`b8b00209 [feat][ops]迁移hstu_v2前向算子`（101 个文件）
> 参与对照的三个仓库坐标见 §1

---

## 0. 结论速览

针对你提出的三个判断，逐条裁定：

| # | 你的判断 | 裁定 | 依据 |
|---|---|---|---|
| 1 | 前向和反向的基础功能互不影响，可以独立迁移 | **成立** | op_host 层前反向完全平行、零交叉 include；op_kernel 分属 `kernel/fwd/` 与 `kernel/bwd/`；TilingData / IN_INDEX / 算子注册均为两份独立复制而非共享引用。见 §5 |
| 2 | 任意 mask 的耦合部分，由于 ops-rec 的反向还没有，可以只单独考虑前向适配 | **不成立** | 上游 RecSDK beta 的反向**早已具备**完整 arbitrary mask 接口：OpDef 已注册 `arbitrary_func` / `sparse_info` 输入、`isArbitrary = (nullptr != arbitraryFunc)`、`predictor_builder.hpp` 已有 `ARBITRARY` 特化。本仓不是「反向还没有」，而是**本仓落后上游**。见 §4 |
| 3 | 有些反向的接口修改，是不是前向其实也被动了 | **大部分不是** | 反向的接口改动分四类来源，其中**只有 2 处**由前向引入（`pipeline_policy.hpp`、`metadata_row_block_scheduler.hpp`，均数值中性）。见 §5.3 |

**一句话总结**：这次迁移的本质不是「加前向、顺带改反向」，而是**把 hstu_v2 整个目录从旧基线（落后上游 RecSDK beta）整体升级到 lihao 分支的状态**。反向被改动的大部分是**版本滞后回填**和**clang-format 重排**，不是前向的副作用；真正由前向牵动的反向改动只有 2 处，且均可证明为中性。

> ⚠ **一处必须知道的例外（复核后新增）**：`block_scheduler.hpp` 的 `GetTriggerSwizzle()` 从「读后清零」改为「粘滞只读」，是**共享层与反向 mainloop 之间的契约变更**。当前 HEAD 两者已同步升级、自洽无影响；但**任何「只换共享文件、保留旧反向 mainloop」的部分迁移路径都会静默算错**（不报错、不崩溃）。详见 **§5.2.1 / N7**。
>
> ✅ **已定路线不触发该危险（2026-09-22 定）**：以 `24b3dd05` 为基线、反向完全不动为标准后，选定的做法是 **B 版 `block_scheduler.hpp` + packed 纯增量** —— 前向对 swizzle API 引用数为 0，`GetTriggerSwizzle()` **一字不改**。详见 §7.2 与 [hstu_v2_forward_migration_plan.md](hstu_v2_forward_migration_plan.md) §3.1。

---

## 1. 三仓关系与版本坐标

| 代号 | 仓库 / 提交 | 日期 | 说明 |
|---|---|---|---|
| **U** | `Ascend/RecSDK` `v26.1.0-beta.0` = `3bcdcac8` | 2026-09-16 | 官方发布基线，注释 `!3165 merge 0916_L1_cache into v26.1.0-beta.0`。前向 + 反向**都完整**，布局 `cust_op/ascendc_op/ai_core_op/hstu_v2/c310/` |
| **B** | 本仓 `24b3dd05` | — | 迁移前的状态，即 `b8b00209` 的**父提交**（`git merge-base --is-ancestor 24b3dd05 b8b00209` = YES）。**B-vs-D 的全部差异 100% 来自 `b8b00209` 这一个 commit** |
| **D** | 本仓当前工作区 | — | 迁移后的状态。内容 == lihao 分支状态 |
| **L** | `lihao_huawei/ops-rec_lihao` `optim_block_sparse_new` = `895f90b` | 2026-09-20 | 迁移来源。与 U **无共同祖先**（merge-base 为空），布局为 `ascendc/attention/` |

### 1.1 lihao 分支上 hstu_v2 的真实提交序列

`1b85a2a`（反向迁入 ops-rec）之后，hstu_v2 只有 **8 个提交**，且集中在 2026-09-19 ~ 09-20：

| 提交 | 日期 | hstu_v2 文件数 | 主题 |
|---|---|---|---|
| `3186a0c` | 09-19 | 0 | Add packed block sparse operator |
| `391c682` | 09-19 | 0 | Add platform-aware HSTU packed scheduling |
| **`06ea2fc`** | 09-19 | **62** | **Optimize HSTU V2 packed sparse pipeline** ← 前向后向在同一个提交里 |
| `7bca6f0` | 09-20 | 8 | Refine HSTU scheduling pipeline |
| `63e5cb6` | 09-20 | 0 | Document HSTU Step6 validation scope |
| `6d81a8f` | 09-20 | 6 | Support varlen packed block sparse inputs |
| `6cf6d50` | 09-20 | 0 | Add msprof HSTU beta comparison report |
| `895f90b` | 09-20 | 2 | Fix packed sparse backward scheduling |

**关键事实**：`06ea2fc` 一个提交同时改了 19 个前向相关文件 + 20 个反向相关文件。**在 lihao 的实际提交历史里，「前向」和「反向」从来不是可分开搬运的两个变更集**——它们是同一次「packed sparse pipeline」重构的两个面。

---

## 2. 文件级全景

### 2.1 集合关系

```
U 的 .hpp/.h/.cpp/CMakeLists 文件数 = 46
B 的                              = 39
D 的                              = 59
```

| 集合 | 数量 | 内容 |
|---|---|---|
| **U 有、D 没有** | **0** | 前向迁移在文件层面**零遗漏** |
| **D 有、U 没有** | 13 | 见下 |
| **B 有、U 没有** | 6 | 全是工程文件（CMakeLists ×3、examples、torch_extension ×2），**内核代码层面 B ⊂ U** |

`D 有、U 没有` 的 13 个：

- 工程/构建：`CMakeLists.txt`、`op_host/CMakeLists.txt`、`op_kernel/CMakeLists.txt`、`examples/aclnn/test_aclnn_hstu_backward_v2.cpp`、`torch_extension/hstu_backward_v2_plugin.cpp`、`torch_extension/hstu_v2_param_check.h` ← **这些 B 也有**（布局差异，非迁移产物）
- 迁移真实新增：`common/hstu_metadata_layout.h`、`common/hstu_schedule_policy.h`、`kernel/pipeline_policy.hpp`、`op_kernel/hstu_forward_v2_packed.cpp`、`op_kernel/hstu_backward_v2_packed.cpp`、`torch_extension/hstu_common.h`、`torch_extension/hstu_forward_v2_plugin.cpp`

### 2.2 内核层面 D 独有（U 完全没有）的真实文件只有 2 个

| 文件 | 行数 | 性质 |
|---|---|---|
| `common/hstu_metadata_layout.h` | 33 | metadata 直接寻址 ABI 常量表（`DIRECT_MAGIC = 0x48535455` 等）。**仅被 `metadata_row_block_scheduler.hpp:37` include** |
| `op_kernel/catlass_hstu/kernel/pipeline_policy.hpp` | 47 | `HstuDirection` 枚举、`HstuPipelinePolicy<Direction, HeadDim>` 前反向各 2 个特化、`HSTU_V2_TASK_TYPE` 宏。**被前向和反向的 builder 共同 include** |

---

## 3. 依赖闭包分析

用脚本沿 `#include "…"` 在仓库内递归计算传递闭包（脚本见 `C:\Users\gg\Desktop\closure.py`）。

### 3.1 前向闭包（入口 `op_kernel/hstu_forward_v2.cpp`）：22 个真实文件

| 文件 | B 有? | 与 U 关系 | 归属 |
|---|---|---|---|
| `detail/target_block_range.hpp` | 无 | U==D | 上游既有，本仓缺 |
| `epilogue/block/block_epilogue_trans_infer.hpp` | 有 | U==D | 三方一致 |
| `epilogue/regbase/common.h` | 有 | U==D | 三方一致 |
| `gemm/block/block_mmad_pv_infer.hpp` | 有 | U==D | 三方一致 |
| `gemm/block/block_mmad_qk_infer.hpp` | 有 | U==D | 三方一致 |
| `gemm/block/block_scheduler_interleaved.hpp` | 有 | U==D | 三方逐字节一致。**前向默认路径的 Q 块调度器**（`USE_INTERLEAVED_Q_SCHEDULER` 生效，交错分核）；B 版即 D 版，无需改动，详见迁移方案 §1.3 |
| `gemm/tile/copy_l0c_to_ub_a5.hpp` | 有 | U==D | 三方一致 |
| `gemm/dispatch_policy.hpp` | 有 | U==D，B 差 3 行 | **纯 clang-format** |
| `gemm/tile/copy_gm_to_l1_a5.hpp` | 有 | U==D，B 差 6 行 | **纯 clang-format** |
| `tla_hstu/layout.hpp` | 有 | U==D，B 差 6 行 | **纯 clang-format** |
| `epilogue/block/block_epilogue_score_infer.hpp` | 有 | D 改 35 行 | 前向专属（反向不用 `*_infer`） |
| `epilogue/regbase/fast_silu_infer.hpp` | 有 | D 改 54 行 | 前向专属 |
| `gemm/block/block_scheduler.hpp` | 有 | **三方全不同** | ⚠ 前反向共享，**含一处契约破点**，见 §5.2.1 |
| `gemm/block/metadata_row_block_scheduler.hpp` | 无 | D 改 74 行 | lihao 新增，前反向共享 |
| `kernel/mask/arbitrary_mask_predictor.hpp` | 无 | D 改 409 行 | lihao 新增，前反向共享 |
| `kernel/mask/no_mask_predictor.hpp` | 有 | D 改 12 行 | **纯加性**，见 §5.3 |
| `kernel/fwd/forward_kernel_{builder,epilogue_mainloop,mmad_mainloop,resource,tiling}.hpp` | 无 | — | 前向专属新增 |
| `op_kernel/hstu_forward_v2.cpp` | 无 | — | 前向专属新增 |

**前向闭包中不存在 `predictor_builder.hpp` 和 `causal_mask_predictor.hpp`**（已 grep 验证，零引用）。前向的 predictor 在 `op_kernel/hstu_forward_v2.cpp:70-71` 直接实例化：

```cpp
using Predictor = std::conditional_t<IS_ARBITRARY, ArbitraryMaskPredictor<BLOCK_M,BLOCK_N,true>,
                                     NoMaskPredictor<BLOCK_M,BLOCK_N>>;
```

**这是一个重要的解耦点**：反向那条路径（走 `PredictorSelector` → `predictor_builder.hpp`）前向根本不经过。

### 3.2 反向闭包（入口 `op_kernel/hstu_backward_v2.cpp`）：28 个真实文件

28 个中 16 个满足 U==D（即本仓只是落后上游），12 个与 U 有差异。

### 3.3 两个闭包的交集：10 个文件

| 共享文件 | B 有? | U-B | U-D | B-D | 判定 |
|---|---|---|---|---|---|
| `detail/target_block_range.hpp` | 无 | — | 0 | — | **纯新增**，B 反向后向不引用 |
| `epilogue/regbase/common.h` | 有 | 0 | 0 | 0 | 三方一致，无需动 |
| `gemm/tile/copy_l0c_to_ub_a5.hpp` | 有 | 0 | 0 | 0 | 三方一致，无需动 |
| `gemm/dispatch_policy.hpp` | 有 | 3 | 0 | 3 | **格式化**，无语义 |
| `gemm/tile/copy_gm_to_l1_a5.hpp` | 有 | 6 | 0 | 6 | **格式化**，无语义 |
| `tla_hstu/layout.hpp` | 有 | 6 | 0 | 6 | **格式化**，无语义 |
| `gemm/block/metadata_row_block_scheduler.hpp` | 无 | — | 74 | — | **纯新增** |
| `kernel/mask/arbitrary_mask_predictor.hpp` | 无 | — | 409 | — | **纯新增** |
| `kernel/mask/no_mask_predictor.hpp` | 有 | 9 | 12 | 21 | **纯加性/中性** |
| `gemm/block/block_scheduler.hpp` | 有 | 154 | 179 | 333 | **加性 opt-in + 1 处契约破点** ⚠ |

**结论：10 个共享文件里，6 个无需任何改动（含 3 个纯格式化），4 个是加性变更。**

⚠ **一处修正**：`block_scheduler.hpp` 的 324 行新增确实是 opt-in 加性的，但 `GetTriggerSwizzle()` 从「读后清零」变「粘滞只读」是 **consumer-side 契约变更**——它与反向 mainloop 必须成套升级。对**已落地的 HEAD**没有影响（反向 mainloop 已同步升级），但**「只换共享文件、保留旧反向 mainloop」的部分迁移路径会静默出错**。详见 §5.2.1。

> **闭包数字更新（2026-09-22）**：上表来自只以 `op_kernel/*.cpp` 为入口的扫描。把入口扩到 `op_host/` 与 `torch_extension/`（plugin 也 include 内核头）后：**前向闭包 31、反向闭包 36、交集 13**。新增进交集的是 `common/hstu_metadata_layout.h`、`kernel/pipeline_policy.hpp`、`torch_extension/hstu_v2_param_check.h`（后者的 B→D 是 +42/-0 纯新增新类）。完整的 A/B/C 三分清单见 [plan §2](hstu_v2_forward_migration_plan.md)。

---

## 4. 影响归因（回答「来源为什么奇怪」）

把 `b8b00209` 对反向的全部改动按**来源**归成四类：

### 类别 ①：上游既有 / 本仓落后 —— 与前向无关

这些文件在 U 里早已存在且与 D 逐字节相同（U-vs-D = 0），本仓只是落后：

| 文件 | U-vs-B |
|---|---|
| `op_host/hstu_backward_v2_define.h` | 13（`IN_INDEX` 补 `Q_SHARE`/`METADATA`/`ARBITRARY_FUNC`/`SPARSE_INFO`） |
| `op_kernel/catlass_hstu/kernel/mask/predictor_builder.hpp` | 43（`GetPredictorType` 支持 `IS_ARBITRARY`、新增 `ARBITRARY` 特化、`PredictorSelector` 扩到 7 参） |
| `gemm/block/block_mmad_dq.hpp` / `_pv.hpp` / `_qk.hpp` | 8 / 12 / 15 |
| `epilogue/block/block_epilogue_rab_grad.hpp` | 18 |
| `epilogue/regbase/fast_rab_grad.hpp` | 6 |
| `op_host/hstu_backward_v2.cpp` 中的 OpDef 三输入注册 | 41 之一 |

**典型实例**：`op_host/hstu_backward_v2.cpp` 的 OpDef 早在上游就把 `metadata` / `arbitrary_func` / `sparse_info` 注册为反向输入了。这不是前向的锅。

**典型实例**：`block_mmad_qk.hpp` 上游新增了 `uint32_t l1ReuseSlot = 0` 参数（**带默认值**）与 L1 复用缓存配套。这是上游 L1 cache 特性的落地，前反向都受影响。

### 类别 ②：纯 clang-format 重排 —— 零语义

| 文件 | 差异内容 |
|---|---|
| `gemm/dispatch_policy.hpp` | 模板参数换行位置 |
| `gemm/tile/copy_gm_to_l1_a5.hpp` | `& dstTensor` 引用风格、`{};` 空体写法、尾注释 |
| `op_kernel/tla_hstu/layout.hpp` | 注释对齐、`}  // namespace` 尾注释 |

`U-vs-B ≠ 0` 但 `U-vs-D == 0` 且差异全是空白/注释 → **这三个文件在报告里看起来像「被改了」，实际一个字符的语义都没变。**

### 类别 ③：lihao 自己的反向功能演进 —— 与前向无关

| 文件 | 内容 |
|---|---|
| `op_host/hstu_backward_v2_tiling.h` +3 | 追加 `maskHeads`、`hasPackedMask` 字段（**追加在结构体末尾，ABI 安全**）+ `REGISTER_TILING_DATA_CLASS(HstuBackwardV2Packed, …)` |
| `op_host/hstu_backward_v2.cpp` +82 | `hasDirectSparse` 检测、`isArbitrary` 语义扩张、packed sparse 容量校验、`class HstuBackwardV2Packed : public HstuBackwardV2` |
| `kernel/bwd/backward_kernel_{mmad,epilogue}_mainloop.hpp` | packed sparse 调度、`SkipBlocks` / `EmptySkipDistance` |
| `kernel/mask/arbitrary_mask_predictor.hpp` +409 | packed state / partial bitmap / `headId` 维 / lazy ApplyMask |
| `epilogue/block/block_epilogue_score_grad.hpp` +63 | 反向的 QK score 梯度 epilogue 适配新 predictor 接口 |
| `epilogue/regbase/fast_silu_grad.hpp` +105 | 反向专属 |
| `op_kernel/hstu_backward_v2_packed.cpp` | 3 行 shim（`#include "hstu_backward_v2.cpp"`） |

**这些是 lihao 给反向加的 packed sparse 功能**，前向并不需要它们。它们和「任意 mask 前向适配」是同一批提交里的两条并行工作。

### 类别 ④：**真由前向引入、反向被动适配** —— 只有 2 处

| 文件 | 变化 | 数值/行为影响 |
|---|---|---|
| `kernel/pipeline_policy.hpp`（新增） | 反向 `backward_kernel_builder.hpp` 的本地 `TileSelector<Element, 128/256>` 被删掉，改为引用 `HstuPipelinePolicy<HstuDirection::BACKWARD, TILE_K>`；反向入口的字面量 `KERNEL_TYPE_MIX_AIC_1_2` 改为宏 `HSTU_V2_TASK_TYPE` | **零**。四个 tile shape 数值（`256,128,128` / `128,64,256`）逐字相同；宏的值与字面量相同 |
| `gemm/block/metadata_row_block_scheduler.hpp`（新增） | 反向 mainloop 改为引用它 | **零**（新增文件，未改反向既有路径） |

---

## 5. 逐文件「不影响反向」可行性验证

### 5.1 基础功能确实互不影响：op_host 层零交叉

| 检查项 | 结论 |
|---|---|
| 反向 host 的 include | 只有 `hstu_backward_v2_define.h` / `hstu_backward_v2_tiling.h` / `bwd/backward_kernel_tiling.hpp` 三条，与前向产物**零交叉** |
| `TilingData` | `HstuBackwardV2TilingData` 与 `HstuForwardV2TilingData` 是两个独立 `BEGIN_TILING_DATA_DEF`，**无共同基类** |
| `IN_INDEX` | 反向 16 项 / 前向 12 项，**两份独立 enum（复制而非引用）** |
| `groups` / `maskHeads` / `hasPackedMask` | 两个 tiling.h 里**各自独立声明**，同名不同体 |
| `op_host/CMakeLists.txt` | **B-vs-D = 0 差异**（用 `file(GLOB ...)` 自动收集） |
| `op_kernel/CMakeLists.txt` | **B-vs-D = 0 差异**（用 `npu_op_kernel_sources(... KERNEL_DIR ./)` glob） |
| `cmake/soc_ops_config.cmake` | 只加了两个**别的算子目录**（`packed_block_sparse`、`hstu_attn_metadata_transformer`）；`hstu_v2` 本就在列 |

### 5.2 `block_scheduler.hpp` —— **加性 opt-in 成立，但含一处行为破点**

> ⚠ **本节曾于初版报告中误判为「完全中性」。经复核，`GetTriggerSwizzle()` 是一处真实的语义变更，会静默破坏「保留 B 版反向 mainloop」的迁移方式。已修正。**

`B-vs-D = 333` 的构成：

- **新增 ~324 行**：`EnablePackedSparseWorkload()`、`EnablePackedSparse()`、`PackedSparseSplitCore()`、`SkipBlocks()`、`IsPackedNonEmpty()`、`GetSwizzlePosition()`/`swizzlePosition`，以及成员 `usePackedSparseWorkload{false}` / `usePackedSparse{false}` / `sparseFirst{true}`
- **删除 9 行**（非注释非空）：

| 删除内容 | 实际情况 | 对 B 反向的影响 |
|---|---|---|
| `bool GetTriggerSwizzle()` | ⚠ **不只是加 `const`，是语义变更**，见下 | **⚠ 破坏，静默精度错** |
| `if (this->batchSize == 1) {` | 变成 `} else if (this->batchSize == 1) {`（前面插了 `useTargetWorkload` / `usePackedSparseWorkload` 分支，均默认 false） | 无（flags 关闭时落到原 `FastSplitCore`） |
| `return (this->blockCnt == 1);` | `return this->blockCnt == 1;` | 无（纯格式化） |
| 6 行注释 | 注释 | 无 |

- **模板参数列表 B 与 D 完全一致**：`RowBlockScheduler<ElementOffset_, BLOCK_M, BLOCK_N>`、`ColumnBlockScheduler<RowBlockScheduler_, BLOCK_M, BLOCK_N, bool USE_SWIZZLE = false, …>` —— **编译层面零障碍**
- **`Init` 重载 B 与 D 一致**：`Init()` + `Init(RowBlockScheduler const&)`

#### 5.2.1 破点：`GetTriggerSwizzle()` 从「读后清零」变成「粘滞只读」

**B 版（`cmp_b/.../block_scheduler.hpp:443-448`）—— 消费式：**

```cpp
bool GetTriggerSwizzle()          // 非 const
{
    auto result = triggerSwizzle;
    triggerSwizzle = false;       // ← 读后清零
    return result;
}
```

B 版**没有** `swizzlePosition` 成员；`Init` 也**不**复位 `triggerSwizzle`（`:415-416` 只在折返时置 true）。

**D 版（`.../block_scheduler.hpp:669-672`）—— 粘滞式：**

```cpp
bool GetTriggerSwizzle() const { return triggerSwizzle; }   // ← 只读，永不消费
```

配套改动（U 上游既有，与本破点成对）：
- `D:633-634` `Init()` 开头新增 `triggerSwizzle = false; swizzlePosition = 0;`
- `D:767 / :773 / :781` 新增 `swizzlePosition` 的递增
- 新增 `GetSwizzlePosition()`（`D:682-684`）
- `L1BlockReuseCache::Probe(blockId, triggerSwizzle, swizzlePosition)` 做「折返窗口 + tag 命中」二次校验

**因果链（已逐环读码确认）：**

| 环节 | 文件:行 | 内容 |
|---|---|---|
| ① 消费点 | `cmp_b/.../kernel/bwd/backward_kernel_mmad_mainloop.hpp:271` | `auto triggerSwizzle = qBlockScheduler.GetTriggerSwizzle();` —— 注意它在 `if (predictor.IsSkip()) continue;`（`:272-274`）**之前**消费 |
| ② 传递 | 同文件 `:276` / `:279` | 作为**第 5 个实参**传给 `blockMmadQK(tQ, tK, pingPongFlag, l0bFlag, triggerSwizzle)` / `blockMmadGV(...)` |
| ③ 落地 | `cmp_b/.../gemm/block/block_mmad_qk.hpp:319` | `if (!triggerSwizzle) { copyGmToL1A(tensorL1a, tensorATile, mSize, kReal, stride); }` —— `true` = **跳过 GM→L1 载入，信任 L1 残留数据** |

**两种语义的实际差别：**

- **B 语义（正确）**：折返遍历中**只有第一个被访问的 Q 块**拿到 `true` → 恰好复用上一次遍历留在 L1 里的那一个块（窗口 = 1 块）
- **D 语义（本身也是正确的）**：`triggerSwizzle` 在整个折返遍历中恒为 `true`，但它的下游是 `Probe()`——由 `swizzlePosition < reuseBlockCount` 限定窗口、由 `cachedBlockIds[slot] == blockId` 做 tag 校验，只有真正命中才返回 `canReuse=true`；且 D 版 mmad 多了 `l1ReuseSlot` 参数（`block_mmad_qk.hpp:290`，带默认值 0）会把命中块放进不同的 L1 子槽位

**破点所在**：若**保留 B 版反向 mainloop** 而换用 D 版 `block_scheduler.hpp`，则 ③ 的 `copyGmToL1A` 跳过判定失去了 `Probe()` 这层校验，退化成「折返遍历里**每一个** Q 块都跳过载入」→ 除首块外全部读到 L1 里别的块的数据 → **反向 `dQ/dK/dV` 静默出错，不报错、不崩溃**。

> **即：`block_scheduler.hpp` 与反向 mainloop 是「成套」的，不能只换其一。** D 版 `block_scheduler.hpp` 是 U 版的严格超集（U-vs-D 为纯新增），因此不能说 D 版「改坏了」反向后端——只能说**它的正确性依赖配套的 `Probe()` 消费者**。

#### 5.2.2 若确需「只换 block_scheduler、保留 B 反向 mainloop」

最小改法是改 B 的反向 mainloop 一行（`backward_kernel_mmad_mainloop.hpp:271`）：

```cpp
auto triggerSwizzle = qBlockScheduler.GetTriggerSwizzle() && qBlockScheduler.GetSwizzlePosition() == 0;
```

**等价性论证**：D 版 `Init()` 把 `swizzlePosition` 清 0，此后每次 `++qBlockScheduler`（含 `continue` 路径——for 的递增子句仍会执行）或 `SkipBlocks()` 都会推进它。所以「`swizzlePosition == 0`」精确对应「本次遍历第一个被访问的块」，与 B 版「读后清零 → 仅首次调用为 true」等价，**含「首块被 predictor 跳过时后续块也不为 true」这一细节**（因为 B:271 在 `IsSkip()` 之前消费）。

> 该等价性为**静态推理结论，未编译验证**；采用时必须用同一组 case 做前后向精度对拍。

#### 5.2.3 对当前仓库状态（D）的影响：**无**

当前 D 的反向 mainloop（`:327-336`）已经是配套写法：

```cpp
auto triggerSwizzle = qBlockScheduler.GetTriggerSwizzle();
auto swizzlePosition = qBlockScheduler.GetSwizzlePosition();
auto qReuse = qReuseCache.Probe(qBlockId, triggerSwizzle, swizzlePosition);
auto gradReuse = gradReuseCache.Probe(qBlockId, triggerSwizzle, swizzlePosition);
...
blockMmadQK(tQ, tK, pingPongFlag, l0bFlag, qReuse.canReuse, qReuse.slot);
```

scheduler 与消费者是**同步升级**的，状态自洽。**本破点只影响「只换共享文件、不动反向 mainloop」的部分迁移路径（即 §7.2 方案 B），不影响已落地的 HEAD。**

> **结论修正：`block_scheduler.hpp` 的 324 行新增确实是 opt-in 加性的（对反向无影响），但那 9 行删除中含一处 consumer-side 契约变更。共享文件与反向 mainloop 必须成套升级。**

### 5.3 `no_mask_predictor.hpp` —— 纯加性

D 相对 B 只做了 4 件事，全部对反向中性：

| 变更 | 性质 |
|---|---|
| 新增 `Construct(Kernel*, const Params&){}` | 空实现；B 的反向从不调用 |
| 新增 `UsesDirectScoreMask() const { return false; }` | 常量返回；B 的反向不调用 |
| 新增 `CopyPackedMask(...){}` | 空实现；B 的反向不调用 |
| `ApplyMask` 返回类型 `void` → `bool`，`return false;` | B 的反向是**语句调用、丢弃返回值** → 源码兼容 |

`MakeBlockPredParams` / `Classifier` / `IsSkip` 三者 B 与 D **逐字节相同**。

### 5.4 `predictor_builder.hpp` / `causal_mask_predictor.hpp` —— 前向不需要，**不要动**

这是最需要注意的地方。这两个文件前向**完全不引用**，所以按「只迁前向」的原则**应该保持 B 版原样**。

> ⚠ **反面警告**：如果把 mask/ 目录整体换成 D 版而保留 B 的反向 mainloop，会**编译通过但运行期崩坏**。原因：U/D 把 predictor 的运行时状态从 `kernel->xxx` 搬到了 `this->xxx` + `Construct()` 注入，而 **B 的反向 mainloop 没有 `Construct()` 调用点**（已 grep 验证 B 全树零调用）。后果：
> - `IS_ARBITRARY=1` 组合 → 解空 `GlobalTensor`
> - `IS_CAUSAL=1 && IS_CONTEXT=1` 组合 → 读空 `gNumContext`
> - `IS_CAUSAL=1 && IS_TARGET=1` 组合 → `causal_mask_predictor.hpp:327` `(i - tbaseQInBlk) / this->params.targetGroupSize` 中 `targetGroupSize` 为成员默认值 **0 → 整数除零**
>
> 即：**mask/ 目录不可单独替换**。要么整个不动（推荐），要么连反向 mainloop + `Params` + tiling 字段 + `block_epilogue_score_grad.hpp` + scheduler 的 `SkipBlocks`/`EnablePackedSparse` 一起搬，并把 `hstu_backward_v2.cpp` 的 `PredictorSelector` 补成 7 参、显式传 `/*IS_FWD=*/false`。

### 5.5 `IS_FWD` 默认值：一个真实的静默陷阱

`predictor_builder.hpp:93-94`：

```cpp
template <bool IS_LOCAL, bool IS_CAUSAL, bool IS_ARBITRARY, uint32_t BLOCK_M, uint32_t BLOCK_N,
          typename ElementOffset = int32_t, bool IS_FWD = true>   // ← 默认 true
using PredictorSelector = …;
```

B 的反向调用点（`cmp_b/op_kernel/hstu_backward_v2.cpp:108`）是 **5 个模板参数**：

```cpp
using Predictor = PredictorSelector<IS_LOCAL, IS_CAUSAL, IS_ARBITRARY, BLOCK_M, BLOCK_N>;
```

而 D 的反向调用点是 7 个，显式传了 `/*IS_FWD=*/false`。

**风险**：若保留 B 的 5 参调用点 + 换 D 的 `predictor_builder.hpp` → `GetPredictorType` 现在会真的返回 `ARBITRARY` → 实例化成 `ArbitraryMaskPredictor<BM, BN, /*IS_FWD=*/true>` → **反向按前向的轴语义寻址，静默算错**。

本仓当前状态（D）**没有这个风险**（反向调用点已显式传 `false`），但如果未来有人做「只换 mask 目录」的局部回退，这是一个必须知道的地雷。

---

## 6. 反向真实变更清单（需要人工确认的部分）

去掉类别 ①②③ 中「与前向无关」的部分，反向**真正因为这次迁移而改变运行时行为**的点只有以下这些：

| # | 变更 | 是否回归 | 说明 |
|---|---|---|---|
| R1 | **删除 2 个 graph 测试用例** | ⚠ **是覆盖回归** | `test_backward_op_precision.py` 10→9、`test_backward_op_precision_mask.py` 6→5，删掉的是 `@pytest.mark.graph`（torch.compile / inductor）用例 |
| R2 | `isArbitrary` 语义扩张 | ⚠ **是行为变更，但仅在新 ABI 面** | `(nullptr != arbitraryFunc)` → `\|\| hasDirectSparse`。B 的 OpDef 根本没有 `sparse_info` 输入，所以**存量调用方不可能触发**；只影响使用了新打包接口的调用方 |
| R3 | `HstuBackwardV2` 与 `HstuBackwardV2Packed` 共用同一个 `TilingFunc` | ⚠ **设计不对称** | 前向把 dense 路径用 `LegacyTilingFunc` 锁住旧行为（强制 `groups/maskHeads/hasPackedMask = 0`），反向**没有这层保护**。建议对齐前向做法 |
| R4 | 反向 packed sparse 能力上线 | ✅ 功能增强 | 反向首次具备真实的 arbitrary mask 支持（迁移前 `isArbitrary` 硬编码 `false`，传任意 mask 会静默算成 dense） |
| R5 | Meta kernel | ✅ **未丢失** | 已核对：`TORCH_LIBRARY_IMPL(ops_rec, Meta, m)` 仍在，且已同步到新 19 参签名。图模式能力保留，只是测试被删（即 R1） |
| R6 | 反向 torch 接口签名 | ✅ 向后兼容 | 尾部追加 `metadata` / `arbitrary_func` / `sparse_info` 三个**带默认值**的可选参数 |
| R7 | 反向 host 侧 ABI（OpDef 输入、kernel 入口 GM_ADDR、`ASCENDC_TPL_SEL_PARAM` 7 参） | ✅ U→D 逐字未变 | B→D 的补全是「上游既有」（类别 ①） |

> **R1 是本次迁移唯一的净损失**，且与「不影响反向」的目标直接相关：反向的图模式（inductor）能力仍在，但失去了自动化回归保护。

---

## 7. 迁移方案

> ⚠ **范围与标准已定（2026-09-22）**：用户明确「反向的既有问题与错漏修复不迁、不关心」，并把标准收紧为 —— **以 `24b3dd05` 为基线，反向完全不动 ∧ 能正常运行 ∧ 功能完备**。据此**方案 A（含反向收口修复）已出局**，且 §7.2 里的 **B-1 / B-2 两条带代价的路线也一并出局**（见下）。
> **可执行的迁移工作令见 [hstu_v2_forward_migration_plan.md](hstu_v2_forward_migration_plan.md)**；本文档 §7 保留为背景与取舍记录。

### 7.1 方案 A（推荐）：当前状态 + 定点收口

当前 `b8b00209` 的落地方式在结构上**已经是「不影响反向」的**——真正的耦合只有 §4 类别 ④ 的 2 处，且均为零数值影响。因此不建议推翻重做，而是做三件收口工作：

**A1. 补回反向图模式测试（消除 R1）**
把 `test_generic_graph` 恢复到 `test_backward_op_precision.py` 和 `test_backward_op_precision_mask.py`。Meta kernel 已验证完好（R5），恢复测试应当直接通过；若因新签名导致 fail，那本身就是需要知道的真实回归。
同时修正 `pytest.ini`：`graph` 标记当前处于「已声明但零使用」状态。

**A2. 对齐前向的 Legacy 保护（消除 R3）**
在 `op_host/hstu_backward_v2.cpp` 中让 `HstuBackwardV2` 走一个 `LegacyTilingKeySet`，强制 `isArbitrary=false`、`hasPackedMask=0`，使 dense op 的旧行为被显式锁死——与前向 `HstuForwardV2 → LegacyTilingFunc` 的做法对称。

**A3. 为 mask/ 目录加一道防线**
在 `predictor_builder.hpp` 的 `PredictorSelector` 处把 `IS_FWD` 的默认值去掉（改为必须显式传），让「反向忘记传 false」在编译期就暴露。

### 7.2 方案 B：严格「只加前向、反向一行不动」

如果目标是**证明反向零改动**（例如需要向评审说明），可以按下面的最小集合重放：

**第一步：纯新增文件（零风险，反向完全不引用）**

```
op_kernel/catlass_hstu/kernel/fwd/*.hpp                     (5 个)
op_kernel/catlass_hstu/kernel/mask/arbitrary_mask_predictor.hpp
op_kernel/catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp
op_kernel/catlass_hstu/gemm/block/l1_block_reuse_cache.hpp
op_kernel/catlass_hstu/detail/target_block_range.hpp
op_kernel/catlass_hstu/kernel/pipeline_policy.hpp
common/hstu_metadata_layout.h
op_kernel/hstu_forward_v2.cpp
op_host/hstu_forward_v2.cpp / _define.h / _tiling.h
torch_extension/hstu_forward_v2_plugin.cpp / hstu_common.h
```

**第二步：前向专属的既有文件（B 也有，但反向不用）**

```
op_kernel/catlass_hstu/epilogue/block/block_epilogue_score_infer.hpp   (反向用的是 _score_grad)
op_kernel/catlass_hstu/epilogue/regbase/fast_silu_infer.hpp            (反向用的是 _silu_grad)
```

**第三步：共享文件，取 D 版**

```
op_kernel/catlass_hstu/kernel/mask/no_mask_predictor.hpp ← 纯加性，对反向零影响，可安全取 D 版（§5.3）
op_kernel/catlass_hstu/gemm/block/block_scheduler.hpp    ← ⚠ 不能单独取，见下
```

⚠ **`block_scheduler.hpp` 单独取 D 版会静默破坏反向**：它与反向 mainloop 是**成套契约**——D 版 `GetTriggerSwizzle()` 从「读后清零」变成「粘滞只读」，正确性依赖消费者改用 `L1BlockReuseCache::Probe()` 做窗口 + tag 校验。B 版反向 mainloop（`backward_kernel_mmad_mainloop.hpp:271`）把该值直接当作「跳过 GM→L1 载入」的开关，换头文件后会退化成「折返遍历里每个 Q 块都跳过载入」→ **`dQ/dK/dV` 静默出错**。详见 §5.2.1。

**但前向的 packed 路径强依赖 D 版**——`EnablePackedSparse` / packed 感知的 `IsValid/IsFirst/IsLast/operator++` 在 U 版和 B 版里都不存在（`forward_kernel_mmad_mainloop.hpp:291`、`forward_kernel_epilogue_mainloop.hpp:272`，均在 `if constexpr (IS_ARBITRARY)` 内）。二者不可兼得，必须在下面两条中选择：

| 选项 | 做法 | 代价 |
|---|---|---|
| **B-1**（推荐） | 取 D 版 `block_scheduler.hpp`，**同时改 B 反向 mainloop 一行**：`GetTriggerSwizzle() && GetSwizzlePosition() == 0`（§5.2.2） | 反向改动 1 行（已论证等价，需精度对拍验证） |
| **B-2** | 取 D 版 `block_scheduler.hpp`，但前向 packed 入口暂不接通 | 反向 0 改动；前向 arbitrary mask 不可用（迁移动机落空） |

> 原「第二步：反向一行不动」的前提在 `block_scheduler.hpp` 上**不成立**——这是本报告初版的错误，已在 §5.2 修正。

#### ⚠ 二次修正（2026-09-22）：B-1 / B-2 **双双出局**

上面「二者不可兼得」的判断**仍然是错的**。真实情况是：前向需要的是 D 版 `block_scheduler.hpp` 增量中的**一个子集**，且该子集与引发契约变更的那部分**零重叠**。

| D 版 `block_scheduler.hpp` 的增量块 | 前向要吗 | 证据 |
|---|---|---|
| packed-sparse（`EnablePackedSparse` / `usePackedSparse` / `sparseFirst` / `FindNextNonEmpty` / `IsPackedNonEmpty` + `Init`/`IsFirst`/`IsLast`/`IsValid` 内的 packed 分支） | **要** | `forward_kernel_{mmad,epilogue}_mainloop.hpp:291/272` |
| `GetTriggerSwizzle()` 语义改写 + `GetSwizzlePosition()` + `swizzlePosition` 状态机 + `l1_block_reuse_cache.hpp` | **不要** | `kernel/fwd/` 全量 grep **0 命中** |
| `EnableTargetWorkload` / `EnablePackedSparseWorkload` / `TargetWorkloadSplitCore` / `GetKBlockWorkload` / `GetTargetQBlockEnd` / `GetSparseRowWorkload` / `PackedSparseSplitCore` 一整套 workload 分核 | **不要** | 调用点全在 D 的反向文件（`kernel/bwd/*:273-280`），B 的反向不调 |
| `RowBlockScheduler` 的 `IsLast()` 去括号、`SplitCore` 三分支改写、被删的注释行 | **不要** | 三分支的两个新条件由上一行的 workload API 驱动 |

**所以正确路线是第三条：B 版 `block_scheduler.hpp` + 只加 packed 增量。**
- `GetTriggerSwizzle()` 保持 B 的「读后清零」，**一字不改** → §5.2.1 的静默算错风险**不触发**
- 反向源码改动 **0 行**，且不需要任何「等价性论证」——不是"等价"，是"没动"
- 剥离可行性已核实：D 版里 `swizzlePosition`（634/684/767/773/781/926）与 `usePackedSparse`/`sparseFirst`（612/654/655/696/697/711/727/761/765/919/920）**零重叠行**，两块功能在源码上不交织

| 选项 | 裁定 |
|---|---|
| B-1 取 D 版 + 改反向 1 行 | **否决** —— 改反向没有必要 |
| B-2 取 D 版 + packed 不接通 | **否决** —— 迁移动机落空 |
| **B-3 ✅ 采纳** | B 版 + packed 纯增量；`usePackedSparse` 恒 `false` ⇒ 反向控制流与 B **逐位相同**。白名单见 [plan §3](hstu_v2_forward_migration_plan.md) |

**第四步：明确不动的文件**

```
op_kernel/catlass_hstu/kernel/mask/predictor_builder.hpp      ← 前向不需（§5.4）
op_kernel/catlass_hstu/kernel/mask/causal_mask_predictor.hpp  ← 前向不需（§5.4）
op_kernel/catlass_hstu/kernel/bwd/*.hpp                       ← 反向主体，**零改动**（B-3 下无 B-1 的那 1 行）
op_host/hstu_backward_v2*.{cpp,h}                             ← 反向 host
torch_extension/hstu_backward_v2_plugin.cpp                   ← 反向插件（除非要 R6 的新接口）
op_kernel/hstu_backward_v2.cpp                                ← 反向入口
```

**注意**：方案 B 会让本仓的 `predictor_builder.hpp` 停留在「`IS_ARBITRARY` 被静默忽略」的旧状态——即**反向传任意 mask 会静默算成 dense**。这是迁移前就存在的既有行为，方案 B 是「保持原样」，不是「引入 bug」。但需要向使用方明确：反向 arbitrary mask 在本方案下**不可用**。

**方案 B 的代价**：本仓共享层与上游 U 分叉度会保持在 43~154 行量级（`predictor_builder.hpp` 43、`causal_mask_predictor.hpp` 66、`block_mmad_dq/pv/qk` 8/12/15 …），后续与上游同步时成本更高。

### 7.3 方案对比

| 维度 | 方案 A（收口） | 方案 B（初版：取 D 版） | **方案 B-3 ✅ 已采纳** |
|---|---|---|---|
| 反向源码改动量 | 少量（R1/R3 修复） | 0（但需 B-1 的 1 行等价论证） | **0（无需任何论证）** |
| 反向运行时行为 | 与上游对齐 | 停留在旧基线 | 停留在旧基线 |
| 反向 arbitrary mask | 可用（R4） | 不可用（静默算 dense） | 不可用（静默算 dense） |
| 共享层改动 | — | 取 D 版 333 行（含 150 行本仓用不到的死代码） | **只加 packed 增量，既有函数体基本不动** |
| 与 lihao `block_scheduler.hpp` 分叉 | 低 | 0 | ~150 行 |
| 适用场景 | 正常迭代 | 需向评审证明「零反向改动」 | **同上，且把保障从「论证」升级为「diff 为空」** |

> **已采纳方案 B-3**（2026-09-22，用户设定标准：以 `24b3dd05` 为基线，反向完全不动 ∧ 能跑 ∧ 功能完备）。
> 方案 A 出局的理由：它要求改反向（R1/R3），与「反向完全不动」直接冲突。方案 B 初版出局的理由：它把「取 D 版再改回 2 处」当成唯一解，而实际只需要 D 版增量的一个子集。
> 代价记录在明处：本仓 `block_scheduler.hpp` 将与 lihao 分叉约 150 行（swizzle + reuse + workload 三块不取）；将来反向若要上 L1 复用，需二次合并——这是把该选择**推迟成独立任务**，而不是现在替反向做决定。

---

## 8. 验证方案

无论选哪个方案，反向的「未被影响」必须由**独立验证**背书，不能只靠论证。

| # | 验证项 | 方法 | 判据 |
|---|---|---|---|
| V1 | 反向精度未回归 | 用 `24b3dd05` 与当前 HEAD 各出一版 `.so`，跑同一组 `test_backward_op_precision*.py`（固定 seed） | 逐 case 数值一致 |
| V2 | 反向 packed/arbitrary 精度 | 跑 `test_backward_op_precision_arbitrary_mask.py`（新增用例） | 达标 |
| V3 | 反向图模式 | 恢复 `test_generic_graph` 后跑 | 构图成功且输出一致 |
| V4 | 前后向一致性 | `test_pipeline_combinations.py` 的四种 pipeline 组合互相对拍 | 四种路径结果 `allclose` |
| V5 | 反向性能未回归 | `run_benchmark.sh` 的 backward 段 + `perf_stats.py` | Task Duration 与基线同量级 |
| V6 | 编译 ABI | 反向 kernel 入口 GM_ADDR 个数、`ASCENDC_TPL_SEL_PARAM` 参数个数与 `24b3dd05` 对比 | 逐字一致 |
| **V7** | **L1 复用（swizzle）路径正确性** | 构造 `USE_SWIZZLE=true` 且同一 `(batch, head)` 出现多轮折返遍历的长序列 case（多 K 块），跑反向精度 | dQ/dK/dV 与 golden 一致。**这是 N7 唯一能暴露的验证手段**——V1 若用短序列可能覆盖不到折返 |

**V1 是核心**：它把「论证」变成「实测」。建议在两个 commit 上各构建一次，用同一份 pytest 参数化集合对拍。

**V7 是 N7 的针对性验证**：折返遍历（同一 `(batch,head)` 被反复访问）在长序列下才出现，短序列 case 可能整轮都不触发 `swizzleDir` 翻转，从而让 N7 类问题漏网。若采用方案 B-1（改 B 反向 mainloop 一行），V7 是**必须做**的等价性对拍。

---

## 9. 遗留风险与不确定项

| # | 风险 | 性质 | 影响 |
|---|---|---|---|
| N1 | `*Packed` op 没有对应的 `ASCENDC_TPL_ARGS_DECL`（前反向都是，`backward_kernel_tiling.hpp` B/D 字节相同） | **推测** | 若 TPL 声明按 opType 索引，`*Packed` 可能取不到模板列表。需实际编译验证 |
| N2 | host 侧硬编码 `BLOCK_M=256 / BLOCK_N=128`（反向，`hstu_backward_v2.cpp:193-194`）与 `HstuPipelinePolicy<BACKWARD,128>::L1TileShape` 前两维必须一致；前向 host 硬编码 `BLOCK=128` 同理 | **已确认存在，无编译期保护** | 改 `HstuPipelinePolicy` 时 host 不报错，只会静默算错 workspace 容量。当前值已逐条比对，一致 |
| N3 | `common/hstu_schedule_policy.h`（`hstu::schedule::WeightedTarget/UniformBoundary`）**全仓零引用** | 已确认 | 悬空占位文件，无害但应知悉 |
| N4 | `isArbitrary` 扩张（R2）的数值等价性 | **推测** | 未做编译/运行验证 |
| N5 | 组合 `IS_CAUSAL=1 && IS_TARGET=1` 下 `IsSkip`/`IsLast` 语义变化（`targetGroupSize` 由假值变真值） | **推测** | lihao 的优化或修正，数值需重过精度 |
| N6 | 分析基于只读静态比对，未编译、未在 NPU 上运行 | 方法论 | §7 的推荐结论依赖 §8 的实测背书 |
| **N7** | **`GetTriggerSwizzle()` 契约变更**：D 版从「读后清零」变「粘滞只读」，与反向 mainloop 成套 | **已确认（读码逐环核实）** | **当前 HEAD 自洽无影响**；但「只换共享文件、保留 B 反向 mainloop」的部分迁移路径会**静默精度错**。详见 §5.2.1 / §7.2 |

**关于 N1 的修正**：`ASCENDC_TPL_ARGS_DECL` 只为非 Packed opType 声明（`HstuForwardV2` / `HstuBackwardV2`），而 `OP_ADD(HstuForwardV2Packed)` / `OP_ADD(HstuBackwardV2Packed)` 又确实注册了 Packed 算子——**前向与反向的写法完全对称，反向已按此模式运行**。因此这不是前向迁移引入的问题，风险等级由「推测/中等」下调为「已知对称模式，随编译验证覆盖即可」。

> **N7 是本次复核中唯一推翻原结论的发现**，也是「不影响反向」这个目标下最需要警惕的一点：**它不会以编译错误的形式暴露**。

---

## 附录 A：复现命令

```bash
# 三方树导出
mkdir -p /tmp/cmp_u /tmp/cmp_b
# U: 上游 RecSDK beta 的 hstu_v2（布局 cust_op/ascendc_op/ai_core_op/hstu_v2/c310/）
# B: 本仓迁移前
git -C <repo> archive 24b3dd05 ascendc/attention/hstu_v2 | tar -x -C /tmp/cmp_b

# 逐文件三方差分
git diff --no-index --numstat <A文件> <B文件>

# 依赖闭包（需将 closure.py 里的 D/U/B 指向实际路径）
python closure.py op_kernel/hstu_forward_v2.cpp
python closure.py op_kernel/hstu_backward_v2.cpp
```

## 附录 B：关键事实速查

| 事实 | 值 |
|---|---|
| 迁移提交 | `b8b00209`，101 个文件（62 新增 / 39 修改 / 0 删除） |
| 迁移前基线 | `24b3dd05` = `b8b00209` 的父提交 |
| 上游对照 | `Ascend/RecSDK` `v26.1.0-beta.0` = `3bcdcac8`（2026-09-16） |
| 迁移来源 | `lihao_huawei/ops-rec_lihao` `optim_block_sparse_new` = `895f90b` |
| 本仓与上游的共同祖先 | **无**（merge-base 为空） |
| U 有 / D 没有的文件 | **0** |
| 前向闭包 | 22 个文件 |
| 反向闭包 | 28 个文件 |
| 前后向共同依赖 | 10 个文件 |
| 其中需改动的 | 4 个（3 个纯加性 + 1 个含契约破点） |
| 真由前向引入的反向改动 | **2 处**（`pipeline_policy.hpp`、`metadata_row_block_scheduler.hpp`，均零数值影响） |
| 反向净损失 | **2 个 graph 测试用例** |
| 唯一推翻初版结论的发现 | **N7**：`GetTriggerSwizzle()` 读后清零 → 粘滞只读（§5.2.1） |
