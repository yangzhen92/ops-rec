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

#ifndef HSTU_GEMM_BLOCK_BLOCK_MMAD_PV_INFER_HPP
#define HSTU_GEMM_BLOCK_BLOCK_MMAD_PV_INFER_HPP

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "buffer_position.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag, class L1TileShape, class L0TileShape, class Element, class Accumulator, uint32_t Stages>
struct ForwardPvMmadBuffer {
    static constexpr uint32_t STAGES = 3U;
    static constexpr uint32_t L1_MK = tla::get<0>(L1TileShape{}) * tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_MN = tla::get<0>(L0TileShape{}) * tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_NK = tla::get<1>(L0TileShape{}) * tla::get<2>(L0TileShape{});
    static constexpr uint32_t L0_MK = tla::get<0>(L0TileShape{}) * tla::get<2>(L0TileShape{});
    using Query = detail::BufferPosition<Element, L1_MK, 0, Stages>;
    using Key = detail::BufferPosition<Element, L0_NK, Query::lowerBound, Stages>;
    using Value = detail::BufferPosition<Element, L0_NK, Key::lowerBound, Stages>;
    using Prob = detail::BufferPosition<Element, L0_MN, Value::lowerBound, STAGES>;
    using Score = detail::BufferPosition<Element, L0_MK / 2U, 0, 2>;
    using UbProb = detail::BufferPosition<Accumulator, L0_MN / 2U, Score::lowerBound, STAGES>;
    using Rab = detail::BufferPosition<Element, L0_MN / 2U, UbProb::lowerBound, 2>;
    using TransOutScratch = detail::BufferPosition<Element, L0_MK / 2U, Rab::lowerBound>;
    using TransIn = detail::BufferPosition<Accumulator, L0_MK / 2U, TransOutScratch::lowerBound, 2>;
    static constexpr uint32_t L1P[STAGES] = {Prob::template Get<0>(), Prob::template Get<1>(), Prob::template Get<2>()};
    static constexpr uint32_t L1V[2] = {Value::template Get<0>(), Value::template Get<1>()};
    static constexpr uint32_t L0A[2] = {0U, ArchTag::L0A_SIZE / 2U};
    static constexpr uint32_t L0B[2] = {0U, ArchTag::L0B_SIZE / 2U};
    static constexpr uint32_t VACC = L0_MN * sizeof(Accumulator);
    static constexpr uint32_t TRANS_OUT_CNT = 2U;
    static constexpr uint32_t TRANS_OUT[2] = {TransIn::template Get<0>(), TransIn::template Get<1>()};
    static_assert(Prob::lowerBound <= ArchTag::L1_SIZE, "forward PV L1 layout exceeds capacity");
    static_assert(TransIn::lowerBound <= ArchTag::UB_SIZE, "forward PV UB layout exceeds capacity");
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool STREAM_K_ATOMIC_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class TileBuffer_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadHSTUPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, 0, false, false, STREAM_K_ATOMIC_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_, ElementC_, TileBuffer_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadHSTUPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, 0, false, false, STREAM_K_ATOMIC_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using TileBuffer = TileBuffer_;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;

    static constexpr uint32_t PROB_STAGES = TileBuffer::STAGES;
    static constexpr bool STREAM_K_ATOMIC = DispatchPolicy::STREAM_K_ATOMIC;

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t headNum, uint32_t headDim,
                 uint32_t const (&V_L1_EVENT_ID_)[2], uint32_t const (&CUBE_CROSS_EVENT_ID_)[PROB_STAGES],
                 uint32_t L0_HANDOFF_ID_, uint32_t L0C_FIX_ID_, uint32_t transReadyId_,
                 uint32_t blockM, uint32_t blockN, uint32_t dimQk)
    {
        detail::ForwardRuntimeBufferLayout<ElementA, ElementAccumulator> layout(blockM, blockN, dimQk, headDim);
        const bool useStaticLayout = blockM == tla::get<0>(L1TileShape_{}) &&
                                     blockN == tla::get<1>(L1TileShape_{});
        for (uint32_t i = 0; i < 2; i++) {
            const uint32_t offset = useStaticLayout ? TileBuffer::L1V[i] : layout.l1.value[i];
            l1VTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(offset);
        }
        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            const uint32_t offset = useStaticLayout ? TileBuffer::L1P[i] : layout.l1.prob[i];
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(offset);
        }

        l0ATensor = resource.l0ABuf.template GetBufferByByte<ElementA>(layout.l0aP);
        l0BTensor = resource.l0BBuf.template GetBufferByByte<ElementB>(layout.l0bV);
        dstTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(layout.l0cPv);

        for (uint32_t i = 0; i < TileBuffer::TRANS_OUT_CNT; i++) {
            const uint32_t offset = useStaticLayout ? TileBuffer::TRANS_OUT[i] : layout.transIn[i];
            OutTensor[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(offset);
        }

        this->stride = headNum * headDim;
        L1_PV_EVENT_ID[0] = V_L1_EVENT_ID_[0];
        L1_PV_EVENT_ID[1] = V_L1_EVENT_ID_[1];

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            cubeReady[i] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[i]);
        }

        L0_HANDOFF_ID = L0_HANDOFF_ID_;
        L0C_FIX_ID = L0C_FIX_ID_;
        transReadyId = transReadyId_;
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    CATLASS_DEVICE void SetStreamKDestination(GM_ADDR dst, uint32_t rowOffset)
    {
        if constexpr (STREAM_K_ATOMIC) {
            if (dst != nullptr) atomicDst.SetGlobalBuffer((__gm__ ElementAccumulator*)dst);
            atomicRowOffset = rowOffset;
        }
    }

    // ============ public API ============

    /// MTE2: DMA V from GM to L1 (pingpong vW/vR).
    template <class TensorSrc>
    CATLASS_DEVICE void LoadGMtoL1(TensorSrc& src)
    {
        nReal = tla::get<0>(src.shape());
        kReal = tla::get<1>(src.shape());
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_PV_EVENT_ID[vW]);
        auto l1Layout = tla::MakeLayout<ElementB, LayoutTagL1B>(nReal, kReal);
        using TileCopy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, typename TileCopy_::TensorL1B>;
        TileCopy{}(tla::MakeTensor(l1VTensor[vW], l1Layout, Arch::PositionL1{}), src, nReal, kReal, stride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_PV_EVENT_ID[vW]);
        vW ^= 1;
    }

    /// MTE1: L1 → L0[1].  Internal flag handshake + cross-core wait.
    template <class Coord>
    CATLASS_DEVICE void LoadL0(Coord const& mn, uint32_t kReal)
    {
        mReal = tla::get<0>(mn);
        nReal = tla::get<1>(mn);
        this->kReal = kReal;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_PV_EVENT_ID[vR]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID);

        LoadL0V(vR, nReal);

        uint32_t pSlot = computeP % PROB_STAGES;
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(cubeReady[pSlot].id);
        LoadL0P(pSlot, nReal);
        computeP++;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_HANDOFF_ID);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_PV_EVENT_ID[vR]);
        vR ^= 1;
    }

    /// CUBE+FIX on L0 → VACC.
    template <class Coord>
    CATLASS_DEVICE void operator()(Coord const& mn, uint32_t kReal, bool isInit, bool isFlush, uint32_t transInSlot)
    {
        mReal = tla::get<0>(mn);
        nReal = tla::get<1>(mn);
        this->kReal = kReal;

        auto l0Coord = tla::MakeCoord(0, 0);
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, nReal);
        auto tensorL0a = tla::MakeTensor(l0ATensor, l0aLayout, l0Coord, Arch::PositionL0A{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nReal, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor, l0bLayout, l0Coord, Arch::PositionL0B{});
        auto l0cLayout = tla::MakeLayoutL0C(mReal, kReal);
        auto tensorL0c = tla::MakeTensor(dstTensor, l0cLayout, l0Coord, Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_HANDOFF_ID);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_ID);

        tileMmad(tensorL0c, tensorL0a, tensorL0b, mReal, kReal, nReal, isInit, 0);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_FIX_ID);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_FIX_ID);

        if (isFlush) {
            if constexpr (STREAM_K_ATOMIC) {
                auto dstLayout = tla::MakeLayout(tla::MakeShape(mReal, kReal),
                                                  tla::MakeStride((int64_t)kReal, tla::Int<1>{}));
                auto tensorDst = tla::MakeTensor(atomicDst, dstLayout,
                                                  tla::MakeCoord((int64_t)atomicRowOffset, 0), Arch::PositionGM{});
                using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<decltype(tensorDst)>;
                AscendC::SetAtomicAdd<ElementAccumulator>();
                CopyL0CToDst{}(tensorDst, tensorL0c, 0);
                AscendC::SetAtomicNone();
            } else {
                FlushL0CToDstPV(mReal, kReal, transInSlot);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_ID);
    }

    // ============ internal helpers ============

    CATLASS_DEVICE void LoadL0V(uint32_t pipeVflag, uint32_t nsize)
    {
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(nsize, kReal);
        auto tensorL1b = tla::MakeTensor(l1VTensor[pipeVflag], l1bLayout, tla::MakeCoord(0, 0), Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nsize, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor, l0bLayout, tla::MakeCoord(0, 0), Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);
    }

    CATLASS_DEVICE void LoadL0P(uint32_t pipeSflag, uint32_t nsize)
    {
        auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mReal, nsize);
        auto tensorL1a = tla::MakeTensor(l1PTensor[pipeSflag], l1aLayout, tla::MakeCoord(0, 0), Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, nsize);
        auto tensorL0a = tla::MakeTensor(l0ATensor, l0aLayout, tla::MakeCoord(0, 0), Arch::PositionL0A{});
        copyL1ToL0A(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void FlushL0CToDstPV(uint32_t mSize, uint32_t nSize, uint32_t transInSlot)
    {
        using TensorL0C = typename TileCopy::TensorL0C;
        auto l0cLayout = tla::MakeLayoutL0C(mSize, nSize);
        auto tensorL0c = tla::MakeTensor(dstTensor, l0cLayout, tla::MakeCoord(0, 0), Arch::PositionL0C{});
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mSize, nSize);
        auto tensorDst = tla::MakeTensor(OutTensor[transInSlot], dstLayout, tla::MakeCoord(0, 0), Arch::PositionUB{});

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, decltype(tensorDst), Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM{}(tensorDst, tensorL0c, 0);

        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(transReadyId);
    }

protected:
    Arch::CrossCoreFlag cubeReady[PROB_STAGES];

    AscendC::LocalTensor<ElementA> l1PTensor[PROB_STAGES];
    AscendC::LocalTensor<ElementB> l1VTensor[2];
    AscendC::LocalTensor<ElementA> l0ATensor;
    AscendC::LocalTensor<ElementB> l0BTensor;
    AscendC::LocalTensor<ElementAccumulator> dstTensor;
    AscendC::LocalTensor<ElementAccumulator> OutTensor[TileBuffer::TRANS_OUT_CNT];
    AscendC::GlobalTensor<ElementAccumulator> atomicDst;
    uint32_t atomicRowOffset{0};

    int64_t stride{0};
    uint32_t mReal{0};
    uint32_t nReal{0};
    uint32_t kReal{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t L1_PV_EVENT_ID[2];
    uint32_t vW = 0, vR = 0;
    uint32_t L0_HANDOFF_ID;
    uint32_t L0C_FIX_ID;
    uint32_t transReadyId;
    uint32_t computeP = 0;
};

}  // namespace Catlass::Gemm::Block

#endif
