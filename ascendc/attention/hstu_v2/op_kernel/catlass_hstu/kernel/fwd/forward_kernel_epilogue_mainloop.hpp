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
 * @file forward_kernel_epilogue_mainloop.hpp
 * @brief HSTU Forward 算子 Epilogue 主循环实现
 * @description 实现 Vector 核上的 Epilogue 操作，包括 SiLU 激活、RAB 相加、
 *              Mask 应用以及 PV 输出转置
 */

#pragma once
#include "forward_events.hpp"
#include "../scheduler_setup.hpp"

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../../../catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp"
#include "../../../catlass_hstu/kernel/mask/arbitrary_mask_predictor.hpp"
#include "../../../catlass_hstu/kernel/mask/no_mask_predictor.hpp"
#include "../../../../common/hstu_schedule_policy.h"

namespace Catlass::Kernel {

/**
 * @brief Forward Epilogue 主循环结构体
 * @tparam BlockEpilogueQK_ QK Epilogue Block
 * @tparam BlockEpiloguePV_ PV Epilogue Block
 * @tparam QBlockScheduler_ Q 方向块调度器
 * @tparam KBlockScheduler_ K 方向块调度器
 * @description 负责在 AI Core 的 Vector 核上执行所有 Epilogue 操作，
 *              包括 SiLU 激活函数、RAB 处理、输出转置等
 */
template <class BlockEpilogueQK_, class BlockEpiloguePV_, class StreamKReduceEpilogue_, class QBlockScheduler_,
          class KBlockScheduler_, class Predictor_, class L1TileShape_, bool IS_LOCAL, bool IS_CONTEXT, bool IS_TARGET,
          bool IS_ARBITRARY, bool ENABLE_STREAM_K = false>
struct ForwardEpilogueMainloop {
    using BlockEpilogueQK = BlockEpilogueQK_;
    using BlockEpiloguePV = BlockEpiloguePV_;
    using StreamKReduceEpilogue = StreamKReduceEpilogue_;
    using QBlockScheduler = QBlockScheduler_;
    using KBlockScheduler = KBlockScheduler_;
    using Predictor = Predictor_;
    using ElementAccumulator = typename BlockEpiloguePV::ElementAccumulator;

    using ArchTag = typename BlockEpilogueQK::ArchTag;
    using ElementQ = typename BlockEpilogueQK::Element;
    // BlockEpilogueScore owns the L0 shape only.  The scheduling geometry is
    // the pipeline L1 shape and is therefore passed explicitly by the kernel
    // builder/entry point.
    using L1TileShape = L1TileShape_;
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});

    static constexpr uint32_t TRANS_READY_ID = ForwardEpilogueEvents::TRANS_READY;
    static constexpr uint32_t TRANS_MTE3_ID = ForwardEpilogueEvents::TRANS_MTE3;
    static constexpr bool IS_ARBITRARY_V = IS_ARBITRARY;
    static constexpr bool IS_LOCAL_V = IS_LOCAL;
    static constexpr bool IS_CAUSAL_V = !IS_LOCAL && BlockEpilogueQK::IS_CAUSAL;
    static constexpr bool IS_CONTEXT_V = IS_CONTEXT;
    static constexpr bool IS_TARGET_V = IS_TARGET;

    struct Params {
        GM_ADDR ptrRab;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;
        GM_ADDR ptrNumContext;
        GM_ADDR ptrNumTarget;
        GM_ADDR ptrAttnOutput;
        GM_ADDR ptrOutputAccum;
        GM_ADDR ptrMetadata;  // 可选: flash_attn_metadata 分核输出;nullptr → 旧设备现算路径
        GM_ADDR ptrSparseInfo;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrRab_, GM_ADDR ptrSeqOffsetQ_, GM_ADDR ptrSeqOffsetK_, GM_ADDR ptrNumContext_,
               GM_ADDR ptrNumTarget_, GM_ADDR ptrAttnOutput_, GM_ADDR ptrOutputAccum_, GM_ADDR ptrMetadata_ = nullptr,
               GM_ADDR ptrSparseInfo_ = nullptr)
            : ptrRab(ptrRab_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_),
              ptrNumContext(ptrNumContext_),
              ptrNumTarget(ptrNumTarget_),
              ptrAttnOutput(ptrAttnOutput_),
              ptrOutputAccum(ptrOutputAccum_),
              ptrMetadata(ptrMetadata_),
              ptrSparseInfo(ptrSparseInfo_)
        {
        }
    };

    CATLASS_DEVICE
    ForwardEpilogueMainloop(GM_ADDR ptrTiling_)
    {
        GET_TILING_DATA(tilingData, ptrTiling_);
        batch = tilingData.batch;
        heads = tilingData.heads;
        dimQK = tilingData.dimQK;
        dimV = tilingData.dimV;
        maxSeqLenQ = tilingData.maxSeqLenQ;
        maxSeqLenK = tilingData.maxSeqLenK;
        blockM = tilingData.blockM;
        blockN = tilingData.blockN;
        totalSeqLenQ = tilingData.totalSeqLenQ;
        targetGroupSize = tilingData.targetGroupSize;
        alpha = tilingData.alpha;
        scale = tilingData.scale;
        windowSizeLeft = tilingData.windowSizeLeft;
        windowSizeRight = tilingData.windowSizeRight;
        rabHeads = tilingData.rabHeads;
        if constexpr (IS_ARBITRARY) {
            maskHeads = tilingData.maskHeads;
        }
        ptrTiling = ptrTiling_;
    }

    template <typename ElementT>
    CATLASS_DEVICE auto MakeTNDTensor(AscendC::GlobalTensor<ElementT>& gt, uint32_t seqLen, uint32_t dim)
    {
        auto layout =
            tla::MakeLayout(tla::MakeShape((int64_t)seqLen * heads, dim), tla::MakeStride(dim, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    template <typename ElementT>
    CATLASS_DEVICE auto MakeBNSSTensor(AscendC::GlobalTensor<ElementT>& gt)
    {
        auto stride0 = (int64_t)rabHeads * maxSeqLenQ * maxSeqLenK;
        auto stride1 = rabHeads == 1 ? 0 : (int64_t)maxSeqLenQ * maxSeqLenK;
        auto layout = tla::MakeLayout(tla::MakeShape(batch, heads, maxSeqLenQ, maxSeqLenK),
                                      tla::MakeStride(stride0, stride1, maxSeqLenK, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    CATLASS_DEVICE
    void InitGlobalTensors(const Params& params, AscendC::GlobalTensor<ElementQ>& gRab,
                           AscendC::GlobalTensor<ElementQ>& gAttnOut)
    {
        gRab.SetGlobalBuffer((__gm__ ElementQ*)params.ptrRab);
        gAttnOut.SetGlobalBuffer((__gm__ ElementQ*)params.ptrAttnOutput);
        if (rabHeads == 1) {
            gRab.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_NORMAL);
        } else {
            gRab.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
    }

    // ============ main pipeline ============

    // transIn 双缓冲（TRANS_IN_BUFFER_CNT=2 恒成立，两种 tile 配置均可驻留 UB）
    // QK-lead-1 模式: QK epilogue 立即处理当前 tile，PV epilogue 延迟一个 Q block
    // 从独立 transIn buffer 排空，避免覆盖
    static constexpr uint32_t TRANS_IN_CNT = BlockEpiloguePV::TileBuffer::TRANS_IN_CNT;

    // 待排空的 PV 输出描述（延迟一个 Q block 后写入 GM）
    struct PvDrainInfo {
        uint32_t qBlockSize = 0;
        uint32_t seqPos = 0;
        uint32_t headIdInBlock = 0;
        uint32_t transInSlot = 0;
    };

    // 排空上一 Q block 的 PV 输出
    template <class GlobalOutput>
    CATLASS_DEVICE void DrainPv(BlockEpiloguePV& blockEpiloguePV, GlobalOutput& gAttnOut, PvDrainInfo const& pv)
    {
        if constexpr (!ENABLE_STREAM_K) {
            auto layout = tla::MakeLayout(tla::MakeShape(pv.qBlockSize, dimV),
                                          tla::MakeStride((int64_t)heads * dimV, tla::Int<1>{}));
            auto tile =
                tla::MakeTensor(gAttnOut, layout, tla::MakeCoord((int64_t)pv.seqPos, (int64_t)pv.headIdInBlock * dimV),
                                Arch::PositionGM{});
            blockEpiloguePV(tile, pv.transInSlot);
        }
    }

    // 推进 scheduler 一步并取下一 tile 坐标（RAB 预取 lookahead）
    // 跨行 next 用局部变量计算，不刷新外层 tAttnOutQ/qBlockSize：
    // 外层变量须保持当前行值供 save 块使用，行末由 save 块统一刷新（bf3671cf 语义）；
    // 返回 false 表示遍历结束
    template <class TensorAttnOut, class TileQ, class CoordNext, class ShapeNext>
    CATLASS_DEVICE bool NextTile(QBlockScheduler& qBlockScheduler, KBlockScheduler& kBlockScheduler,
                                 TensorAttnOut const& tensorAttnOut, TileQ& tAttnOutQ, uint32_t& qBlockSize,
                                 CoordNext& coordNext, ShapeNext& shapeNext)
    {
        ++kBlockScheduler;
        if (!kBlockScheduler.IsValid()) {
            ++qBlockScheduler;
            if (!qBlockScheduler.IsValid())
                return false;
            kBlockScheduler.Init(qBlockScheduler);
            auto tAttnOutQNext = qBlockScheduler.GetTile(tensorAttnOut);
            auto qBlockSizeNext = tla::get<0>(tAttnOutQNext.shape());
            auto tV = kBlockScheduler.GetTile(tensorAttnOut);
            auto nextMapping = kBlockScheduler.GetTileMapping(tAttnOutQNext.coord(), tAttnOutQNext.shape());
            coordNext = tla::get<0>(nextMapping);
            shapeNext = tla::MakeShape(qBlockSizeNext, tla::get<0>(tV.shape()));
            return true;
        }
        auto tV = kBlockScheduler.GetTile(tensorAttnOut);
        auto nextMapping = kBlockScheduler.GetTileMapping(tAttnOutQ.coord(), tAttnOutQ.shape());
        coordNext = tla::get<0>(nextMapping);
        shapeNext = tla::MakeShape(qBlockSize, tla::get<0>(tV.shape()));
        return true;
    }

    template <class TensorRab, class TensorAttnOut, class GlobalOutput>
    CATLASS_DEVICE void RunPipeline(BlockEpilogueQK& blockEpilogueQK, BlockEpiloguePV& blockEpiloguePV,
                                    QBlockScheduler& qBlockScheduler, KBlockScheduler& kBlockScheduler,
                                    TensorRab& tensorRab, TensorAttnOut& tensorAttnOut, GlobalOutput& gAttnOut,
                                    Predictor& predictor)
    {
        auto tQ0 = qBlockScheduler.GetTile(tensorAttnOut);
        auto tV0 = kBlockScheduler.GetTile(tensorAttnOut);
        auto mapping0 = kBlockScheduler.GetTileMapping(tQ0.coord(), tQ0.shape());
        auto coord0 = tla::get<0>(mapping0);
        auto shape0 = tla::MakeShape(tla::get<0>(tQ0.shape()), tla::get<0>(tV0.shape()));
        blockEpilogueQK.LoadRab(tensorRab, coord0, shape0);

        auto tAttnOutQ = qBlockScheduler.GetTile(tensorAttnOut);
        uint32_t qBlockSize = tla::get<0>(tAttnOutQ.shape());

        auto coordNext = coord0;
        auto shapeNext = shape0;
        bool hasNext = true;
        bool hasPv = false;
        uint32_t transInSlot = 0;
        PvDrainInfo savedPv;

        while (qBlockScheduler.IsValid()) {
            while (kBlockScheduler.IsValid()) {
                bool isFinal = kBlockScheduler.IsLast();

                auto coord = coordNext;
                auto shape = shapeNext;
                auto meta = qBlockScheduler.GetMeta();
                auto predParams = predictor.MakeBlockPredParams(
                    tla::MakeCoord((uint32_t)tla::get<0>(meta), (uint32_t)tla::get<1>(meta),
                                   (uint32_t)qBlockScheduler.GetHeadBlockId(), (uint32_t)kBlockScheduler.GetBlockId()),
                    this, qBlockScheduler.GetCurrentSeqLen(), kBlockScheduler.GetSeqLens(), 1);
                predictor.Classifier(predParams);
                hasNext = NextTile(qBlockScheduler, kBlockScheduler, tensorAttnOut, tAttnOutQ, qBlockSize, coordNext,
                                   shapeNext);
                blockEpilogueQK(tensorRab, coord, shape, coordNext, shapeNext, hasNext, predictor);

                if (hasPv) {
                    DrainPv(blockEpiloguePV, gAttnOut, savedPv);
                    hasPv = false;
                }

                if (isFinal && !ENABLE_STREAM_K) {
                    auto blockOffset = static_cast<uint32_t>(tla::get<0>(tAttnOutQ.coord()));
                    savedPv = {qBlockSize, blockOffset / heads, blockOffset % heads, transInSlot};
                    transInSlot = (transInSlot + 1) % TRANS_IN_CNT;
                    hasPv = true;
                    tAttnOutQ = qBlockScheduler.GetTile(tensorAttnOut);
                    qBlockSize = tla::get<0>(tAttnOutQ.shape());
                }
            }
        }

        if (hasPv && !ENABLE_STREAM_K) {
            DrainPv(blockEpiloguePV, gAttnOut, savedPv);
        }
    }

    CATLASS_DEVICE
    void operator()(const Params& params)
    {
        AscendC::GlobalTensor<ElementQ> gRab;
        AscendC::GlobalTensor<ElementQ> gAttnOut;
        InitGlobalTensors(params, gRab, gAttnOut);
        AscendC::GlobalTensor<float> gOutputAccum;
        if constexpr (ENABLE_STREAM_K)
            gOutputAccum.SetGlobalBuffer((__gm__ float*)params.ptrOutputAccum);

        auto tensorRab = MakeBNSSTensor(gRab);
        auto tensorAttnOut = MakeTNDTensor(gAttnOut, totalSeqLenQ, dimV);

        // 行(Q)调度器: 经工厂构造,对 OuterBlockScheduler / InterleavedOuterBlockScheduler /
        // MetadataOuterBlockScheduler 统一(见 mmad mainloop 注释)。
        QBlockScheduler qBlockScheduler = Gemm::Block::MakeRowScheduler<QBlockScheduler>(
            batch, heads, params.ptrSeqOffsetQ, params.ptrSeqOffsetK, params.ptrMetadata, blockM, blockN);
        KBlockScheduler kBlockScheduler(batch, heads, params.ptrSeqOffsetK, blockM, blockN);
        InitializeForwardSchedulers<IS_ARBITRARY, IS_LOCAL, BlockEpilogueQK::IS_CAUSAL, IS_CONTEXT, IS_TARGET,
                                    ENABLE_STREAM_K, BLOCK_M, BLOCK_N>(
            qBlockScheduler, kBlockScheduler, params, maskHeads, maxSeqLenQ, maxSeqLenK, windowSizeLeft,
            windowSizeRight, targetGroupSize, blockM, blockN);
        const bool hasWork = kBlockScheduler.IsValid() && qBlockScheduler.IsValid();

        const uint32_t qkReadyEvents[3] = {ForwardEpilogueEvents::QK_READY_0, ForwardEpilogueEvents::QK_READY_1,
                                           ForwardEpilogueEvents::QK_READY_2};
        const uint32_t pvReadyEvents[3] = {ForwardEpilogueEvents::PV_READY_0, ForwardEpilogueEvents::PV_READY_1,
                                           ForwardEpilogueEvents::PV_READY_2};
        BlockEpilogueQK blockEpilogueQK(qkReadyEvents, pvReadyEvents, resource, alpha, scale, blockM, blockN, dimQK,
                                        dimV);
        BlockEpiloguePV blockEpiloguePV(TRANS_READY_ID, TRANS_MTE3_ID, resource, scale, blockM, blockN, dimQK, dimV);
        Predictor predictor;
        predictor.Construct(this, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(TRANS_MTE3_ID);
        if (hasWork) {
            if constexpr (ENABLE_STREAM_K)
                RunPipeline(blockEpilogueQK, blockEpiloguePV, qBlockScheduler, kBlockScheduler, tensorRab,
                            tensorAttnOut, gOutputAccum, predictor);
            else
                RunPipeline(blockEpilogueQK, blockEpiloguePV, qBlockScheduler, kBlockScheduler, tensorRab,
                            tensorAttnOut, gAttnOut, predictor);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BlockEpilogueQK::RAB_MTE2_V_ID[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BlockEpilogueQK::RAB_MTE2_V_ID[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(TRANS_MTE3_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BlockEpilogueQK::SCORE_MTE3_V_ID[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BlockEpilogueQK::SCORE_MTE3_V_ID[1]);
        if constexpr (ENABLE_STREAM_K) {
            // Pairwise Cube -> Vector completion precedes the vector-wide
            // barrier; otherwise reduction can observe a still-zero or only
            // partially accumulated outputAccum buffer.
            // One completion token is emitted by the paired AIC.  Consume it
            // from one AIV only, then publish the observation to its sibling
            // through the following AIV-wide rendezvous.
            if (AscendC::GetSubBlockIdx() == 0) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(TRANS_READY_ID);
            }
            AscendC::SyncAll();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            uint32_t coreId = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum() * AscendC::GetTaskRatio();
            const auto range = hstu::schedule::UniformCoreRange(totalSeqLenQ, coreId, coreNum);
            const uint32_t rows = range.endBlock - range.beginBlock;
            if (rows != 0U) {
                auto srcLayout =
                    tla::MakeLayout(tla::MakeShape(rows, dimV), tla::MakeStride((int64_t)dimV, tla::Int<1>{}));
                auto src = tla::MakeTensor(gOutputAccum, srcLayout, tla::MakeCoord((int64_t)range.beginBlock, 0),
                                           Arch::PositionGM{});
                auto dstLayout = tla::MakeLayout(tla::MakeShape(totalSeqLenQ, heads, dimV),
                                                 tla::MakeStride((int64_t)heads * dimV, (int64_t)dimV, tla::Int<1>{}));
                auto dst = tla::MakeTensor(gAttnOut, dstLayout, tla::MakeCoord((int64_t)range.beginBlock, 0, 0),
                                           Arch::PositionGM{});
                // Forward score generation intentionally leaves 1/C out of
                // the low-precision P tile. Apply it once, after all split-K
                // FP32 partials have been accumulated.
                StreamKReduceEpilogue reduce(resource, scale);
                reduce.template operator()<true>(dst, src);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    Arch::Resource<ArchTag> resource;

    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimV{0};
    uint32_t maxSeqLenQ{0};
    uint32_t maxSeqLenK{0};
    uint32_t totalSeqLenQ{0};
    uint32_t targetGroupSize{0};
    int32_t windowSizeLeft{-1}, windowSizeRight{-1};
    uint32_t blockM{BLOCK_M}, blockN{BLOCK_N};
    uint32_t rabHeads{0};
    float alpha{0.0f};
    float scale{0.0f};
    uint32_t maskHeads{0};
    GM_ADDR ptrTiling{nullptr};
};

}  // namespace Catlass::Kernel
