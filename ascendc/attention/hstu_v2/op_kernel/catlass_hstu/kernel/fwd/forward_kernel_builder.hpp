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

#ifndef HSTU_CATLASS_FORWARD_KERNEL_BUILDER_HPP
#define HSTU_CATLASS_FORWARD_KERNEL_BUILDER_HPP

#include "../../../tla_hstu/layout.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_gm_to_l1_a5.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_l0c_to_ub_a5.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_qk_infer.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_pv_infer.hpp"
#include "../../../catlass_hstu/gemm/block/block_scheduler.hpp"
#include "../../../catlass_hstu/gemm/block/block_scheduler_interleaved.hpp"
#include "../../../catlass_hstu/gemm/block/stream_k_scheduler.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_score_infer.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_trans_infer.hpp"
#include "../../../catlass_hstu/epilogue/block/stream_k_reduce_epilogue.hpp"
#include "../../../catlass_hstu/kernel/pipeline_policy.hpp"

// =============================================================================
// 分核策略切换 (硬编码，重新编译生效)
//   注释掉 → 原始连续分块 (OuterBlockScheduler)
//   取消注释 → 交替分块 (InterleavedOuterBlockScheduler)，提高 L2 Cache 跨核共享率
// =============================================================================
#define USE_INTERLEAVED_Q_SCHEDULER

// =============================================================================
// K/V Cluster 数量：调参开关
//   数值越大 → L2 冲突越分散，但跨核共享率越低
//   数值越小 → 共享率越高，但冲突可能增加
//   可通过编译选项覆盖：-DHSTU_FWD_KV_CLUSTER_NUM=2
// =============================================================================
#ifndef HSTU_FWD_KV_CLUSTER_NUM
#define HSTU_FWD_KV_CLUSTER_NUM 4
#endif

using namespace Catlass;
using namespace tla;

namespace Catlass::Kernel {

template <typename ArchTag_, typename ElementType_, typename ElementOffset_, uint32_t TILE_K_, bool HAS_RAB_,
          bool IS_LOCAL_, bool IS_CAUSAL_, bool IS_ARBITRARY_>
struct ForwardKernelConfig {
    using ElementType = ElementType_;
    using ElementOffset = ElementOffset_;
    using ArchTag = ArchTag_;

    static constexpr uint32_t TILE_K = TILE_K_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool IS_LOCAL = IS_LOCAL_;
    static constexpr bool IS_CAUSAL = IS_CAUSAL_;
    static constexpr bool HAS_MASK = IS_LOCAL || IS_CAUSAL || IS_ARBITRARY_;
    static constexpr bool IS_ARBITRARY = IS_ARBITRARY_;

    using PipelinePolicy = HstuPipelinePolicy<HstuDirection::FORWARD, TILE_K>;
    using L1TileShape = typename PipelinePolicy::L1TileShape;
    using L0TileShape = typename PipelinePolicy::L0TileShape;
    static constexpr uint32_t STAGES = 2;
};

template <typename ArchTag_, typename ElementType_, typename L1TileShape_,
          typename L0TileShape_, bool HAS_RAB_, bool HAS_MASK_, bool IS_LOCAL_, bool IS_CAUSAL_, bool IS_ARBITRARY_>
struct QKBlockBuilder {
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;
    static constexpr bool IS_LOCAL = IS_LOCAL_;
    static constexpr bool IS_CAUSAL = IS_CAUSAL_;
    static constexpr bool IS_ARBITRARY = IS_ARBITRARY_;
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementQ = ElementType;
    using LayoutQ = layout::zN;
    using ElementK = ElementType;
    using LayoutK = layout::nZ;
    using ElementS = typename Gemm::helper::ElementAccumulatorSelector<ElementQ, ElementK>::ElementAccumulator;
    // Score masking is row-oriented.  Keep QK output and PV input in
    // RowMajor whenever either RAB or a mask is present; using zN with
    // HAS_MASK makes the compact per-row bitmap address the wrong lanes once
    // a sequence spans more than one K block.
    using LayoutS = std::conditional_t<HAS_RAB || HAS_MASK, layout::RowMajor, layout::zN>;

    using TileCopyTlaQK =
        std::conditional_t<HAS_RAB,
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                             LayoutS, void, Gemm::Tile::CopyL0CToUBMode::RESERVED>,
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                             LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;

    using DispatchPolicy = Gemm::MmadHSTUQK<ArchTag, false, false>;

    using MmadTileBuffer = Gemm::Block::ForwardQkMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementQ, ElementS, 2U>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS,
                                                MmadTileBuffer, TileCopyTlaQK>;

    using EpilogueTileBuffer = Epilogue::Block::ForwardScoreEpilogueBuffer<
        ArchTag, L1TileShape, L0TileShape, ElementQ, ElementS, 2U>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueScore<ArchTag, ElementQ, ElementS, EpilogueTileBuffer,
                                                              L0TileShape, HAS_RAB, HAS_MASK, IS_LOCAL,
                                                              IS_CAUSAL, IS_ARBITRARY>;
};

template <typename ArchTag_, typename ElementType_, typename L1TileShape_,
          typename L0TileShape_, bool HAS_RAB_, bool HAS_MASK_, bool ENABLE_STREAM_K_>
struct PVBlockBuilder {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    using ElementP = ElementType;
    using LayoutP = std::conditional_t<HAS_RAB || HAS_MASK, layout::RowMajor, layout::zN>;
    using ElementV = ElementType;
    using LayoutV = layout::zN;
    using ElementO = typename Gemm::helper::ElementAccumulatorSelector<ElementP, ElementV>::ElementAccumulator;
    using LayoutO = layout::RowMajor;

    using TileCopyTlaPV = std::conditional_t<ENABLE_STREAM_K_,
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementP, LayoutP, ElementV, LayoutV, ElementO, LayoutO>,
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutP, ElementV, LayoutV, ElementO, LayoutO, void,
                                         Gemm::Tile::CopyL0CToUBMode::RESERVED>>;

    using DispatchPolicy = Gemm::MmadHSTUPV<ArchTag, false, false, 0, false, false, ENABLE_STREAM_K_>;

    using MmadTileBuffer = Gemm::Block::ForwardPvMmadBuffer<ArchTag, L1TileShape, L0TileShape, ElementP, ElementO, 2U>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementP, ElementV, ElementO,
                                                MmadTileBuffer, TileCopyTlaPV>;

    using EpilogueTileBuffer = Epilogue::Block::ForwardTransEpilogueBuffer<
        ArchTag, L0TileShape, ElementV, ElementO>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueTransOut<
        ArchTag, EpilogueTileBuffer, Epilogue::Block::TransTag::UB_TO_GM, ElementO, ElementV, HAS_RAB>;
};

template <typename ArchTag_, typename ElementOffset_, typename L1TileShape_, bool SERPENTINE_ = false>
struct BlockSchedulerBuilder {
    using ArchTag = ArchTag_;
    using ElementOffset = ElementOffset_;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});

#ifdef USE_INTERLEAVED_Q_SCHEDULER
    using QBlockScheduler = Gemm::Block::InterleavedOuterBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N, SERPENTINE_>;
#else
    using QBlockScheduler = Gemm::Block::OuterBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>;
#endif
    using KBlockScheduler =
        Gemm::Block::InnerBlockScheduler<QBlockScheduler, BLOCK_M, BLOCK_N, false, false, HSTU_FWD_KV_CLUSTER_NUM>;
};

template <typename KernelConfig_, bool USE_CAUSAL_SERPENTINE_ = false, bool ENABLE_STREAM_K_ = false>
struct ForwardKernelBuilder {
    using Config = KernelConfig_;

    using ElementType = typename Config::ElementType;
    using ElementOffset = typename Config::ElementOffset;
    using ArchTag = typename Config::ArchTag;

    static constexpr bool HAS_RAB = Config::HAS_RAB;
    static constexpr bool HAS_MASK = Config::HAS_MASK;
    static constexpr bool IS_CAUSAL = Config::IS_CAUSAL;
    static constexpr int32_t TILE_K = Config::TILE_K;
    static constexpr uint32_t STAGES = Config::STAGES;

    using AccumulatorElementType =
        typename Gemm::helper::ElementAccumulatorSelector<ElementType, ElementType>::ElementAccumulator;

    // Tile shapes
    using L1TileShape = typename Config::L1TileShape;
    using L0TileShape = typename Config::L0TileShape;

    // Block QK Builder
    using QKBlockBuilder = QKBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape, HAS_RAB,
                                          HAS_MASK, Config::IS_LOCAL, IS_CAUSAL, Config::IS_ARBITRARY>;
    using BlockMmadQK = typename QKBlockBuilder::BlockMmad;
    using BlockEpilogueQK = typename QKBlockBuilder::BlockEpilogue;

    // Block PV Builder
    using PVBlockBuilder =
        PVBlockBuilder<ArchTag, ElementType, L1TileShape, L0TileShape, HAS_RAB, HAS_MASK, ENABLE_STREAM_K_>;
    using BlockMmadPV = typename PVBlockBuilder::BlockMmad;
    using BlockEpiloguePV = typename PVBlockBuilder::BlockEpilogue;
    using StreamKBuffer = Epilogue::Block::ForwardStreamKReduceBuffer<
        ArchTag, L0TileShape, ElementType, AccumulatorElementType>;
    using StreamKReduceEpilogue = Epilogue::Block::StreamKReduceEpilogue<
        ArchTag, StreamKBuffer, AccumulatorElementType, ElementType>;

    using DefaultSchedulerBuilder =
        BlockSchedulerBuilder<ArchTag, ElementOffset, L1TileShape, USE_CAUSAL_SERPENTINE_>;
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    using StreamOuter = Gemm::Block::StreamKOuterBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>;
    using StreamInner = Gemm::Block::StreamKInnerBlockScheduler<StreamOuter, BLOCK_M, BLOCK_N>;
    using QBlockScheduler = std::conditional_t<ENABLE_STREAM_K_, StreamOuter,
                                               typename DefaultSchedulerBuilder::QBlockScheduler>;
    using KBlockScheduler = std::conditional_t<ENABLE_STREAM_K_, StreamInner,
                                               typename DefaultSchedulerBuilder::KBlockScheduler>;
};

}  // namespace Catlass::Kernel

#endif
