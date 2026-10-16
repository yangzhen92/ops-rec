/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

/**
 • @file backward_kernel_builder.hpp

 • @brief HSTU Backward 算子 Kernel 构建器

 • @description 提供各种 Kernel 配置的构建器模板，用于组装完整的Backward算子内核

 •              包含: Tile选择器、Kernel配置、Block构建器、调度器构建器等

 */

#pragma once

#include "../../../tla_hstu/layout.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_gm_to_l1_a5.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_l0c_to_ub_a5.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_qk.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_pv.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_dq.hpp"
#include "../../../catlass_hstu/gemm/block/block_scheduler.hpp"
#include "../../../catlass_hstu/gemm/block/stream_k_scheduler.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_score_grad.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_rab_grad.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_trans_out.hpp"
#include "../../../catlass_hstu/epilogue/block/stream_k_reduce_epilogue.hpp"
#include "../../../catlass_hstu/kernel/pipeline_policy.hpp"

using namespace Catlass;
using namespace tla;

namespace Catlass::Kernel {

/**
 • @brief Tile 选择器模板

 • @tparam Element 数据元素类型

 • @tparam TILE_K K 方向的 Tile 大小

 • @description 根据 TILE_K 大小选择不同的 L1/L0 Tile 形状配置

 */
/**
 • @brief Backward Kernel 配置结构体

 • @tparam ArchTag_ 架构标签

 • @tparam ElementType_ 数据类型

 • @tparam ElementOffset_ 序列偏移量类型

 • @tparam TILE_K_ K 方向 Tile 大小

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 聚合所有 Kernel 配置信息，包括 Tile 形状、数据类型、开关标志等

 */
template <typename ArchTag_, typename ElementType_, typename ElementOffset_, uint32_t TILE_K_, bool HAS_RAB_,
          bool IS_LOCAL_, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY>
struct BackwardKernelConfig {
    using ElementType = ElementType_;
    using ElementOffset = ElementOffset_;
    using ArchTag = ArchTag_;

    static constexpr uint32_t TILE_K = TILE_K_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool IS_LOCAL = IS_LOCAL_;
    static constexpr bool HAS_MASK = IS_LOCAL_ || IS_CAUSAL || IS_ARBITRARY;
    static constexpr bool IS_ARBITRARY_MASK = IS_ARBITRARY;
    static constexpr bool DIRECT_CAUSAL_MASK = IS_CAUSAL && !IS_LOCAL_ && !IS_CONTEXT && !IS_TARGET && !IS_ARBITRARY;

    using PipelinePolicy = HstuPipelinePolicy<HstuDirection::BACKWARD, TILE_K>;
    using L1TileShape = typename PipelinePolicy::L1TileShape;
    using L0TileShape = typename PipelinePolicy::L0TileShape;
    static constexpr uint32_t STAGES = 2;
};

/**
 • @brief QK Block 构建器

 • @tparam ArchTag_ 架构标签

 • @tparam ElementType_ 数据类型

 • @tparam TileBufferType_ 缓冲区类型模板

 • @tparam L1TileShape_ L1 Tile 形状

 • @tparam L0TileShape_ L0 Tile 形状

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 构建 Q * K^T 矩阵乘法所需的 Block MMAD 组件，包括:

 •              - 数据类型和布局定义

 •              - Tile 拷贝策略

 •              - MMAD 策略选择

 •              - Block 调度器类型

 */
template <typename ArchTag_, typename ElementType_, typename L1TileShape_, typename L0TileShape_, bool HAS_RAB_,
          bool HAS_MASK_, bool IS_LOCAL_, bool IS_ARBITRARY_, bool DIRECT_CAUSAL_>
struct QKBlockBuilder {
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;
    static constexpr bool IS_LOCAL = IS_LOCAL_;
    static constexpr bool IS_ARBITRARY = IS_ARBITRARY_;
    static constexpr bool DIRECT_CAUSAL = DIRECT_CAUSAL_;
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    // layout type define
    using ElementQ = ElementType;
    using LayoutQ = layout::zN;
    using ElementK = ElementType;
    using LayoutK = layout::nZ;
    using ElementS = typename Gemm::helper::ElementAccumulatorSelector<ElementQ, ElementK>::ElementAccumulator;
    using LayoutS = std::conditional_t<(HAS_RAB || HAS_MASK), layout::RowMajor, layout::zN>;

    static constexpr auto CopyMode =
        (HAS_RAB || HAS_MASK) ? Gemm::Tile::CopyL0CToUBMode::RESERVED : Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    using TileCopyTla =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS, void,
                                          CopyMode, false, Gemm::Tile::ScaleGranularity::PER_TENSOR>;

    using DispatchPolicy = Gemm::MmadHSTUQK<ArchTag, false, false, true>;

    using MmadTileBuffer = Gemm::Block::BackwardQkMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementQ, ElementS, 2U>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS,
                                                MmadTileBuffer, TileCopyTla>;

    using EpilogueTileBuffer = Epilogue::Block::BackwardScoreEpilogueBuffer<ArchTag, L1TileShape, L0TileShape, ElementQ,
                                                                            ElementS, 2U, HAS_RAB>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogueScoreGrad<ArchTag, ElementQ, ElementS, EpilogueTileBuffer, L0TileShape,
                                                L1TileShape, HAS_RAB, HAS_MASK, IS_LOCAL, IS_ARBITRARY, DIRECT_CAUSAL>;
};

/**
 • @brief GV Block 构建器

 • @tparam ArchTag_ 架构标签

 • @tparam ElementType_ 数据类型

 • @tparam TileBufferType_ 缓冲区类型模板

 • @tparam L1TileShape_ L1 Tile 形状

 • @tparam L0TileShape_ L0 Tile 形状

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 构建 Attention Score * V (P * V) 矩阵乘法所需的 Block MMAD 组件

 */
template <typename ArchTag_, typename ElementType_, typename L1TileShape_, typename L0TileShape_,
          class BlockEpilogueQK_>
struct GVBlockBuilder {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using BlockEpilogueQK = BlockEpilogueQK_;
    static constexpr bool HAS_RAB = BlockEpilogueQK::HAS_RAB;
    static constexpr bool HAS_MASK = BlockEpilogueQK::HAS_MASK;

    // layout type define
    using ElementG = ElementType;
    using LayoutG = layout::zN;
    using ElementV = ElementType;
    using LayoutV = layout::nZ;
    using ElementGS = typename Gemm::helper::ElementAccumulatorSelector<ElementG, ElementV>::ElementAccumulator;
    using LayoutGS = std::conditional_t<(HAS_RAB || HAS_MASK), layout::RowMajor, layout::zN>;

    using TileCopyTla =
        std::conditional_t<(HAS_RAB || HAS_MASK),
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementG, LayoutG, ElementV, LayoutV, ElementGS,
                                                             LayoutGS, void, Gemm::Tile::CopyL0CToUBMode::RESERVED,
                                                             false, Gemm::Tile::ScaleGranularity::PER_TENSOR>,
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementG, LayoutG, ElementV, LayoutV, ElementGS,
                                                             LayoutGS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                                                             false, Gemm::Tile::ScaleGranularity::PER_TENSOR>>;

    using DispatchPolicy = Gemm::MmadHSTUQK<ArchTag, false, false, true>;

    using MmadTileBuffer =
        Gemm::Block::BackwardGvMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementG, ElementGS, 2U>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementG, ElementV, ElementGS,
                                                MmadTileBuffer, TileCopyTla>;

    using EpilogueTileBuffer =
        Epilogue::Block::BackwardRabGradEpilogueBuffer<ArchTag, L1TileShape, L0TileShape, ElementG, ElementGS, 2U>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueRabGrad<BlockEpilogueQK, EpilogueTileBuffer>;
};

/**
 • @brief KV Grad Block 构建器

 • @tparam ArchTag_ 架构标签

 • @tparam ElementType_ 数据类型

 • @tparam TileBufferType_ 缓冲区类型模板

 • @tparam L1TileShape_ L1 Tile 形状

 • @tparam L0TileShape_ L0 Tile 形状

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 构建 dV 和 dK 梯度计算所需的 Block MMAD 组件

 */
template <typename ArchTag_, typename ElementType_, typename L1TileShape_, typename L0TileShape_, bool HAS_RAB_,
          bool ENABLE_STREAM_K_>
struct KVGradBlockBuilder {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    static constexpr bool HAS_RAB = HAS_RAB_;

    // layout type define
    using ElementP = ElementType;
    using LayoutP = layout::nZ;
    using ElementQ = ElementType;
    using LayoutG = layout::zN;

    using ElementGrab = ElementType;
    using ElementG = ElementType;

    using ElementXGrad = ElementType;  // X means K or V
    using ElementXGradAcc = typename Gemm::helper::ElementAccumulatorSelector<ElementP, ElementG>::ElementAccumulator;
    using LayoutXGrad = layout::RowMajor;

    using DispatchPolicyVGrad = Gemm::MmadHSTUPV<ArchTag, false, false, 0, false, true, ENABLE_STREAM_K_>;
    using VGradTileCopyTla = std::conditional_t<
        ENABLE_STREAM_K_,
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementP, LayoutP, ElementG, LayoutG, ElementXGradAcc, LayoutXGrad>,
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutP, ElementG, LayoutG, ElementXGradAcc, LayoutXGrad,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;

    // GEMM VGrad
    using VGradTileBuffer = Gemm::Block::BackwardKvGradMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementType,
                                                                  ElementXGradAcc, 2U, false>;
    using BlockMmadVGrad = Gemm::Block::BlockMmadTla<DispatchPolicyVGrad, L1TileShape, L0TileShape, ElementP, ElementG,
                                                     ElementXGradAcc, VGradTileBuffer, VGradTileCopyTla>;

    using DispatchPolicyKGrad = Gemm::MmadHSTUPV<ArchTag, false, false, 0, false, true, ENABLE_STREAM_K_>;
    using KGradTileCopyTla = std::conditional_t<
        ENABLE_STREAM_K_,
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementP, LayoutP, ElementG, LayoutG, ElementXGradAcc, LayoutXGrad>,
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutP, ElementG, LayoutG, ElementXGradAcc, LayoutXGrad,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;
    // GEMM KGrad
    using KGradTileBuffer = Gemm::Block::BackwardKvGradMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementType,
                                                                  ElementXGradAcc, 2U, true>;
    using BlockMmadKGrad = Gemm::Block::BlockMmadTla<DispatchPolicyKGrad, L1TileShape, L0TileShape, ElementGrab,
                                                     ElementQ, ElementXGradAcc, KGradTileBuffer, KGradTileCopyTla>;

    using EpilogueTileBuffer =
        Epilogue::Block::BackwardKvTransEpilogueBuffer<ArchTag, L0TileShape, ElementXGrad, ElementXGradAcc, 2U>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogueTransOut<ArchTag, EpilogueTileBuffer, Epilogue::Block::TransTag::UB_TO_GM,
                                               ElementXGradAcc, ElementXGrad, HAS_RAB, true>;
};

/**
 • @brief Q Grad Block 构建器

 • @tparam ArchTag_ 架构标签

 • @tparam ElementType_ 数据类型

 • @tparam TileBufferType_ 缓冲区类型模板

 • @tparam L1TileShape_ L1 Tile 形状

 • @tparam L0TileShape_ L0 Tile 形状

 • @description 构建 dQ 梯度计算所需的 Block MMAD 组件 (dO * K^T)

 */
template <typename ArchTag_, typename ElementType_, typename L1TileShape_, typename L0TileShape_>
struct QGradBlockBuilder {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    // layout type define
    using ElementGrab = ElementType;
    using LayoutGrab = layout::zN;
    using ElementK = ElementType;
    using LayoutK = layout::zN;
    using ElementQGrad = typename Gemm::helper::ElementAccumulatorSelector<ElementGrab, ElementK>::ElementAccumulator;
    using LayoutQGrad = layout::RowMajor;

    using DispatchPolicy = Gemm::MmadHSTUDQ<ArchTag, false, false>;

    using TileCopyTla =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementGrab, LayoutGrab, ElementK, LayoutK, ElementQGrad, LayoutQGrad>;

    using MmadTileBuffer =
        Gemm::Block::BackwardDqMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementGrab, ElementQGrad, 2U>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementGrab, ElementK,
                                                ElementQGrad, MmadTileBuffer, TileCopyTla>;

    using EpilogueTileBuffer = Epilogue::Block::BackwardQTransEpilogueBuffer<ArchTag, ElementGrab, ElementQGrad, 2U>;
    using BlockEpilogue =
        Epilogue::Block::StreamKReduceEpilogue<ArchTag, EpilogueTileBuffer, ElementQGrad, ElementGrab>;
};

/**
 • @brief Block 调度器构建器

 • @tparam ArchTag_ 架构标签

 • @tparam ElementOffset_ 序列偏移量类型

 • @tparam L1TileShape_ L1 Tile 形状

 • @description 构建行方向和列方向的 Block 调度器，用于多核任务分配

 */
template <typename ArchTag_, typename ElementOffset_, typename L1TileShape_>
struct BlockSchedulerBuilder {
    using ArchTag = ArchTag_;
    using ElementOffset = ElementOffset_;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});

    using KBlockScheduler = Gemm::Block::OuterBlockScheduler<ElementOffset, BLOCK_N, BLOCK_M>;
    using QBlockScheduler = Gemm::Block::InnerBlockScheduler<KBlockScheduler, BLOCK_N, BLOCK_M, true, false, 1, true>;
};

/**
 • @brief Backward Kernel 构建器

 • @tparam KernelConfig_ Kernel 配置类型

 • @description 整合所有 Block 构建器，构建完整的 Backward 算子 Kernel

 •              包含 QK、GV、KVGrad、QGrad 四个 Block 构建器以及调度器

 */
template <typename KernelConfig_, bool ENABLE_STREAM_K_ = false>
struct BackwardKernelBuilder {
    using Config = KernelConfig_;

    using ElementType = typename Config::ElementType;
    using ElementOffset = typename Config::ElementOffset;
    using ArchTag = typename Config::ArchTag;

    static constexpr bool HAS_RAB = Config::HAS_RAB;
    static constexpr bool HAS_MASK = Config::HAS_MASK;
    static constexpr int32_t TILE_K = Config::TILE_K;
    static constexpr uint32_t STAGES = Config::STAGES;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementType, ElementType>::ElementAccumulator;

    // Tile shapes
    using L1TileShape = typename Config::L1TileShape;
    using L0TileShape = typename Config::L0TileShape;

    // Block QK Builder
    using QKBlockBuilder = QKBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape, HAS_RAB, HAS_MASK,
                                          Config::IS_LOCAL, Config::IS_ARBITRARY_MASK, Config::DIRECT_CAUSAL_MASK>;
    using BlockMmadQK = typename QKBlockBuilder::BlockMmad;
    using BlockEpilogueQK = typename QKBlockBuilder::BlockEpilogue;

    // Block GV Builder
    using GVBlockBuilder = GVBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape, BlockEpilogueQK>;
    using BlockMmadGV = typename GVBlockBuilder::BlockMmad;
    using BlockEpilogueGV = typename GVBlockBuilder::BlockEpilogue;

    // Block KVGrad Builder
    using KVGradBlockBuilder =
        KVGradBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape, HAS_RAB, ENABLE_STREAM_K_>;
    using BlockMmadVGrad = typename KVGradBlockBuilder::BlockMmadVGrad;
    using BlockMmadKGrad = typename KVGradBlockBuilder::BlockMmadKGrad;
    using BlockEpilogueKVGrad = typename KVGradBlockBuilder::BlockEpilogue;

    // Block QGrad Builder
    using QGradBlockBuilder = QGradBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape>;
    using BlockMmadQGrad = typename QGradBlockBuilder::BlockMmad;
    using BlockEpilogueQGrad = typename QGradBlockBuilder::BlockEpilogue;

    using DefaultSchedulerBuilder = BlockSchedulerBuilder<ArchTag, ElementOffset, L1TileShape>;
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    using StreamOuter = Gemm::Block::StreamKOuterBlockScheduler<ElementOffset, BLOCK_N, BLOCK_M>;
    using StreamInner = Gemm::Block::StreamKInnerBlockScheduler<StreamOuter, BLOCK_N, BLOCK_M, true>;
    using KBlockScheduler =
        std::conditional_t<ENABLE_STREAM_K_, StreamOuter, typename DefaultSchedulerBuilder::KBlockScheduler>;
    using QBlockScheduler =
        std::conditional_t<ENABLE_STREAM_K_, StreamInner, typename DefaultSchedulerBuilder::QBlockScheduler>;
};

}  // namespace Catlass::Kernel
