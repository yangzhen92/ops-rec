/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "../../gemm/block/buffer_position.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag, class L0TileShape, class Element, class Accumulator>
struct ForwardStreamKReduceBuffer {
    static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM =
        tla::get<0>(L0TileShape{}) * tla::get<2>(L0TileShape{}) / 2U;
    static constexpr uint32_t L0_MN = tla::get<0>(L0TileShape{}) * tla::get<1>(L0TileShape{});
    using Score = Gemm::Block::detail::BufferPosition<Element, TILE_Q_GRAD_MAX_ELEM, 0, 2>;
    using Prob = Gemm::Block::detail::BufferPosition<Accumulator, L0_MN / 2U, Score::lowerBound, 3>;
    using Rab = Gemm::Block::detail::BufferPosition<Element, L0_MN / 2U, Prob::lowerBound, 2>;
    using TransOutScratch = Gemm::Block::detail::BufferPosition<Element, TILE_Q_GRAD_MAX_ELEM, Rab::lowerBound>;
    using TransIn = Gemm::Block::detail::BufferPosition<Accumulator, TILE_Q_GRAD_MAX_ELEM, TransOutScratch::lowerBound, 2>;
    static constexpr uint32_t STAGES = 2U;
    static constexpr uint32_t TRANS_IN[STAGES] = {TransIn::template Get<0>(), TransIn::template Get<1>()};
    static constexpr uint32_t TRANS_OUT[STAGES] = {Score::template Get<0>(), Score::template Get<1>()};
    static_assert(TransIn::lowerBound <= ArchTag::UB_SIZE, "forward Stream-K layout exceeds UB capacity");
};

// Final reduction epilogue shared by dQ and Stream-K O/dK/dV.  The source is
// fp32 head-major [H,T,D] storage and the destination is TND [T,H,D].
template <class ArchTag_, class TileBuffer_, class ElementAccumulator_, class Element_>
class StreamKReduceEpilogue {
public:
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using Element = Element_;
    using TileBuffer = TileBuffer_;
    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t TILE_MAX_ELEM = TileBuffer::TILE_Q_GRAD_MAX_ELEM;
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE explicit StreamKReduceEpilogue(Arch::Resource<ArchTag>& resource,
                                                   ElementAccumulator scale = static_cast<ElementAccumulator>(1))
        : scale(scale)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            ubIn[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::TRANS_IN[i]);
            ubOut[i] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::TRANS_OUT[i]);
        }
    }

    template <bool APPLY_SCALE = false, class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst& dst, TensorSrc& src)
    {
        const uint32_t rows = tla::get<0>(src.shape());
        const uint32_t totalRows = tla::get<0>(dst.shape());
        const uint32_t dim = tla::get<1>(src.shape());
        const uint32_t heads = tla::get<1>(dst.shape());
        if (rows == 0U || dim == 0U) return;
        AscendC::DataCopyParams params;
        params.blockLen = dim / ELE_NUM_PER_C0;
        params.srcStride = 0;
        params.dstStride = (tla::get<0>(dst.stride()) - tla::get<1>(dst.stride())) / ELE_NUM_PER_C0;
        uint32_t slot = 0;
        const uint32_t rowsPerLoop = TILE_MAX_ELEM / dim;
        const uint32_t loops = (rows + rowsPerLoop - 1U) / rowsPerLoop;
        for (uint32_t h = 0; h < heads; ++h) {
            int64_t srcOffset = src.layout()(src.coord()) + h * totalRows * dim;
            int64_t dstOffset = dst.layout()(dst.coord()) + h * dim;
            for (uint32_t loop = 0; loop < loops; ++loop) {
                const uint32_t begin = loop * rowsPerLoop;
                const uint32_t count = min(rowsPerLoop, rows - begin);
                params.blockCount = count;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
                AscendC::DataCopy(ubIn[slot], src.data()[srcOffset], count * dim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
                if constexpr (APPLY_SCALE) {
                    AscendC::Muls(ubIn[slot], ubIn[slot], scale, count * dim);
                }
                AscendC::Cast(ubOut[slot], ubIn[slot], AscendC::RoundMode::CAST_RINT, count * dim);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
                AscendC::DataCopy(dst.data()[dstOffset], ubOut[slot], params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
                slot ^= 1U; srcOffset += count * dim; dstOffset += count * tla::get<0>(dst.stride());
            }
        }
        // A following output reduction reuses the same ping-pong UB. Drain
        // both outstanding UB-to-GM transfers before handing ownership back
        // to the caller, then restore the initial tokens for the next reduce.
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

private:
    ElementAccumulator scale{static_cast<ElementAccumulator>(1)};
    AscendC::LocalTensor<ElementAccumulator> ubIn[STAGES];
    AscendC::LocalTensor<Element> ubOut[STAGES];
};

}  // namespace Catlass::Epilogue::Block
