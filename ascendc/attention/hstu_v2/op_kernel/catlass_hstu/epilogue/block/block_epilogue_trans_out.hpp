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
 * @file block_epilogue_trans_out.hpp
 * @brief HSTU Epilogue 输出转置 Block 实现
 * @description 提供数据从 UB 转存到 GM 的操作，
 *              用于将计算结果从 Unified Buffer 写回到 Global Memory
 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "../../../catlass_hstu/epilogue/regbase/common.h"
#include "../../gemm/block/buffer_position.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag, class L0TileShape, class Element, class Accumulator, uint32_t Stages>
struct BackwardKvTransEpilogueBuffer {
    static constexpr uint32_t L0_MN = tla::get<0>(L0TileShape{}) * tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_NK = tla::get<1>(L0TileShape{}) * tla::get<2>(L0TileShape{});
    using Score = Gemm::Block::detail::BufferPosition<Accumulator, L0_MN>;
    using TransV = Gemm::Block::detail::BufferPosition<Accumulator, L0_NK / 2U, Score::lowerBound>;
    using TransOutBuffer = Gemm::Block::detail::BufferPosition<Element, L0_NK / 2U, TransV::lowerBound>;
    using Prob = Gemm::Block::detail::BufferPosition<Element, L0_MN, Score::lowerBound>;
    using Grab =
        Gemm::Block::detail::BufferPosition<Element, L0_MN,
                                            (TransOutBuffer::lowerBound > Prob::lowerBound ? TransOutBuffer::lowerBound
                                                                                           : Prob::lowerBound)>;
    using GrabPart = Gemm::Block::detail::BufferPosition<Element, L0_MN, Grab::lowerBound>;
    using Gs = Gemm::Block::detail::BufferPosition<Accumulator, L0_MN, GrabPart::lowerBound>;
    static constexpr uint32_t MASK_END = Gs::lowerBound + tla::get<0>(L0TileShape{}) * 64U;
    static constexpr uint32_t TRANS_K_OFFSET = tla::get<2>(L0TileShape{}) == 128U ? GrabPart::upperBound : MASK_END;
    using TransK = Gemm::Block::detail::BufferPosition<Accumulator, L0_NK / 2U, TRANS_K_OFFSET>;
    static constexpr uint32_t TRANS_IN_CNT = 2U;
    static constexpr uint32_t TRANS_IN[2] = {TransV::template Get<0>(), TransK::template Get<0>()};
    static constexpr uint32_t TRANS_OUT = TransOutBuffer::template Get<0>();
    static_assert(TransK::lowerBound <= ArchTag::UB_SIZE, "backward KV transpose layout exceeds UB capacity");
};

template <class ArchTag, class Element, class Accumulator, uint32_t Stages>
struct BackwardQTransEpilogueBuffer {
    static constexpr uint32_t ELEM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM =
        RoundDown(ArchTag::UB_SIZE / Stages / (sizeof(Accumulator) + sizeof(Element)), ELEM_PER_C0);
    using TransIn = Gemm::Block::detail::BufferPosition<Accumulator, TILE_Q_GRAD_MAX_ELEM, 0, Stages>;
    using TransOutBuffer =
        Gemm::Block::detail::BufferPosition<Element, TILE_Q_GRAD_MAX_ELEM, TransIn::lowerBound, Stages>;
    static constexpr uint32_t STAGES = Stages;
    static constexpr uint32_t TRANS_IN[Stages] = {TransIn::template Get<0>(), TransIn::template Get<1>()};
    static constexpr uint32_t TRANS_OUT[Stages] = {TransOutBuffer::template Get<0>(),
                                                   TransOutBuffer::template Get<1>()};
    static_assert(TransOutBuffer::lowerBound <= ArchTag::UB_SIZE, "backward Q transpose layout exceeds UB capacity");
};

/**
 * @brief 转置类型枚举
 * @description UB_TO_GM: 从 UB 写回 GM
 */
enum class TransTag : uint8_t {
    UB_TO_GM = 0
};

/**
 * @brief Epilogue 输出转置 Block
 * @tparam ArchTag_ 架构标签
 * @tparam TileBuffer_ Tile 缓冲区类型
 * @tparam TransTag_ 转置类型
 * @tparam ElementAccumulator_ 累加器数据类型
 * @tparam Element_ 目标数据类型
 * @tparam HAS_RAB 是否有相对位置偏置
 * @description 执行计算结果从 UB 到 GM 的转存操作
 */
template <class ArchTag_, class TileBuffer_, TransTag TransTag_, class ElementAccumulator_,
          class Element_ = ElementAccumulator_, bool HAS_RAB = false, bool SAME_AB = false>
struct BlockEpilogueTransOut {
    static_assert(DEPENDENT_FALSE<ArchTag_>, "BlockEpilogueTransOut specialization failed!.");
};

template <class ArchTag_, class TileBuffer_, class ElementAccumulator_, class Element_, bool HAS_RAB, bool SAME_AB>
struct BlockEpilogueTransOut<ArchTag_, TileBuffer_, TransTag::UB_TO_GM, ElementAccumulator_, Element_, HAS_RAB,
                             SAME_AB> {
public:
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using Element = Element_;
    using TileBuffer = TileBuffer_;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    BlockEpilogueTransOut(uint32_t cubeFlag, uint32_t ubFreeFlag, uint32_t stride, Arch::Resource<ArchTag>& resource,
                          uint32_t transInSlot, ElementAccumulator scale = static_cast<ElementAccumulator>(1))
    {
        // AIC addresses AIV0/AIV1 with global flag IDs base/base+FLAG_MAX.
        // Each AIV waits on the same base ID in its own local namespace.
        this->cubeReady = Arch::CrossCoreFlag(cubeFlag);
        this->ubFree = Arch::CrossCoreFlag(ubFreeFlag);
        this->stride = stride;
        this->scale = scale;
        this->ubTransIn =
            resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::TRANS_IN[transInSlot]);
        this->ubTransOut = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::TRANS_OUT);
    }

    template <bool APPLY_SCALE = false, class TensorA>
    CATLASS_DEVICE void operator()(TensorA& dstTensor)
    {
        constexpr uint32_t SUB_BLOCK_MASK = SAME_AB ? 0x4 : 0x2;
        AscendC::CrossCoreWaitFlag<SUB_BLOCK_MASK, PIPE_V>(cubeReady.id);

        auto totalRows = tla::get<0>(dstTensor.shape());
        auto cols = tla::get<1>(dstTensor.shape());
        auto rowsPerAiv = CeilDiv<2>(totalRows);
        auto rowOffset = AscendC::GetSubBlockIdx() * rowsPerAiv;
        auto rows =
            rowOffset < totalRows ? (rowsPerAiv < totalRows - rowOffset ? rowsPerAiv : totalRows - rowOffset) : 0U;
        auto dstOffset = dstTensor.layout()(dstTensor.coord()) + rowOffset * stride;

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = rows;
        intriParams.blockLen = cols / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = (stride - cols) / ELE_NUM_PER_C0;

        if (rows > 0) {
            // The two FP32 inputs ping-pong, but both conversions reuse one
            // low-precision output. Wait for the preceding UB->GM copy before
            // overwriting that output buffer.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            Convert<APPLY_SCALE>(rows * cols);
            // Input ownership ends after the vector load/conversion, so AIC
            // may refill this slot while MTE3 drains the shared output.
            AscendC::CrossCoreSetFlag<SUB_BLOCK_MASK, PIPE_V>(ubFree.id);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopy(dstTensor.data()[dstOffset], ubTransOut, intriParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            AscendC::CrossCoreSetFlag<SUB_BLOCK_MASK, PIPE_V>(ubFree.id);
        }
    }

    template <class TensorA>
    CATLASS_DEVICE void StoreZero(TensorA& dstTensor)
    {
        auto totalRows = tla::get<0>(dstTensor.shape());
        auto cols = tla::get<1>(dstTensor.shape());
        auto rowsPerAiv = CeilDiv<2>(totalRows);
        auto rowOffset = AscendC::GetSubBlockIdx() * rowsPerAiv;
        auto rows =
            rowOffset < totalRows ? (rowsPerAiv < totalRows - rowOffset ? rowsPerAiv : totalRows - rowOffset) : 0U;
        if (rows == 0) {
            return;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord()) + rowOffset * stride;
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = rows;
        intriParams.blockLen = cols / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = (stride - cols) / ELE_NUM_PER_C0;

        // Empty sparse K tiles have no AIC producer. Reuse the epilogue output
        // UB directly, without consuming or returning any cross-core token.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::Duplicate<Element>(ubTransOut, static_cast<Element>(0), rows * cols);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(dstTensor.data()[dstOffset], ubTransOut, intriParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

private:
    template <bool APPLY_SCALE>
    CATLASS_DEVICE void Convert(uint32_t count)
    {
        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));
        auto src = (__ubuf__ ElementAccumulator*)ubTransIn.GetPhyAddr();
        auto dst = (__ubuf__ Element*)ubTransOut.GetPhyAddr();
        AscendC::VF_CALL<ConvertVf<APPLY_SCALE>>(src, dst, scale, count, repeatTimes);
    }

    template <bool APPLY_SCALE>
    __simd_vf__ static inline void ConvertVf(__ubuf__ ElementAccumulator* src, __ubuf__ Element* dst,
                                             ElementAccumulator scale, uint32_t count, uint32_t repeatTimes)
    {
        constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(ElementAccumulator);
        AscendC::MicroAPI::RegTensor<ElementAccumulator> value;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::RegTensor<ElementAccumulator> scaleReg;
        if constexpr (APPLY_SCALE) {
            AscendC::MicroAPI::Duplicate(scaleReg, scale);
        }
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<ElementAccumulator>(count);
            AscendC::MicroAPI::LoadAlign(value, src + i * ONE_REP_ELM);
            if constexpr (APPLY_SCALE) {
                AscendC::MicroAPI::Mul(value, value, scaleReg, mask);
            }
            if constexpr (!std::is_same<Element, ElementAccumulator>::value) {
                Catlass::Epilogue::RegBase::CastDownStore<Element, ElementAccumulator>(dst + i * ONE_REP_ELM, value,
                                                                                       mask);
            } else {
                AscendC::MicroAPI::StoreAlign(dst + i * ONE_REP_ELM, value, mask);
            }
        }
    }

    int64_t stride{0};
    ElementAccumulator scale{static_cast<ElementAccumulator>(1)};
    Arch::CrossCoreFlag cubeReady;
    Arch::CrossCoreFlag ubFree;

    AscendC::LocalTensor<ElementAccumulator> ubTransIn;
    AscendC::LocalTensor<Element> ubTransOut;
};

}  // namespace Catlass::Epilogue::Block
