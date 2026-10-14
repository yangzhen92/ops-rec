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
 • @file block_epilogue_trans_infer.hpp

 • @brief HSTU Epilogue 输出转置 Block 实现

 • @description 提供数据从 UB 转存到 GM 的操作

 •              用于将计算结果从 Unified Buffer 写回到 Global Memory

 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "../../gemm/block/buffer_position.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag, class L0TileShape, class Element, class Accumulator>
struct ForwardTransEpilogueBuffer {
    static constexpr uint32_t PHYSICAL_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t PHYSICAL_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_MN = tla::get<0>(L0TileShape{}) * tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_MK = tla::get<0>(L0TileShape{}) * tla::get<2>(L0TileShape{});
    using Score = Gemm::Block::detail::BufferPosition<Element, L0_MK / 2U, 0, 2>;
    using Prob = Gemm::Block::detail::BufferPosition<Accumulator, L0_MN / 2U, Score::lowerBound, 3>;
    using Rab = Gemm::Block::detail::BufferPosition<Element, L0_MN / 2U, Prob::lowerBound, 2>;
    using TransOutScratch = Gemm::Block::detail::BufferPosition<Element, L0_MK / 2U, Rab::lowerBound>;
    using TransIn = Gemm::Block::detail::BufferPosition<Accumulator, L0_MK / 2U, TransOutScratch::lowerBound, 2>;
    static constexpr uint32_t TRANS_IN_CNT = 2U;
    static constexpr uint32_t TRANS_IN[2] = {TransIn::template Get<0>(), TransIn::template Get<1>()};
    static constexpr uint32_t TRANS_OUT = TransOutScratch::template Get<0>();
    static_assert(TransIn::lowerBound <= ArchTag::UB_SIZE, "forward transpose layout exceeds UB capacity");
};

/**
 • @brief 转置类型枚举

 • @description UB_TO_GM: 从 UB 写回 GM

 */
enum class TransTag : uint8_t {
    UB_TO_GM = 0
};

/**
 • @brief Epilogue 输出转置 Block

 • @tparam ArchTag_ 架构标签

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @tparam TransTag_ 转置类型

 • @tparam ElementAccumulator_ 累加器数据类型

 • @tparam Element_ 目标数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @description 执行计算结果从 UB 到 GM 的转存操作

 */
template <class ArchTag_, class TileBuffer_, TransTag TransTag_, class ElementAccumulator_,
          class Element_ = ElementAccumulator_, bool HAS_RAB = false>
struct BlockEpilogueTransOut {
    static_assert(DEPENDENT_FALSE<ArchTag_>, "BlockEpilogueTransOut specialization failed!.");
};

template <class ArchTag_, class TileBuffer_, class ElementAccumulator_, class Element_, bool HAS_RAB>
struct BlockEpilogueTransOut<ArchTag_, TileBuffer_, TransTag::UB_TO_GM, ElementAccumulator_, Element_, HAS_RAB> {
public:
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using Element = Element_;
    using TileBuffer = TileBuffer_;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    BlockEpilogueTransOut(uint32_t cubeFlag, uint32_t transEventId, Arch::Resource<ArchTag>& resource,
                          ElementAccumulator scale_, uint32_t blockM, uint32_t blockN,
                          uint32_t dimQk, uint32_t dimV)
    {
        Gemm::Block::detail::ForwardRuntimeBufferLayout<Element, ElementAccumulator> layout(
            blockM, blockN, dimQk, dimV);
        const bool useStaticLayout = blockM == TileBuffer::PHYSICAL_M && blockN == TileBuffer::PHYSICAL_N;
        this->cubeReady = Arch::CrossCoreFlag(cubeFlag);
        this->transEventId = transEventId;
        this->scale = scale_;
        this->ubTransOut = resource.ubBuf.template GetBufferByByte<Element>(
            useStaticLayout ? TileBuffer::TRANS_OUT : layout.transOut);
        for (uint32_t i = 0; i < TileBuffer::TRANS_IN_CNT; i++) {
            const uint32_t offset = useStaticLayout ? TileBuffer::TRANS_IN[i] : layout.transIn[i];
            this->ubTransIn[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(offset);
        }
    }

    template <class TensorA>
    CATLASS_DEVICE void operator()(TensorA& dstTensor, uint32_t transInSlot = 0)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(cubeReady.id);

        auto coreId = AscendC::GetSubBlockIdx();
        auto totalRows = tla::get<0>(dstTensor.shape());
        auto rowsCore0 = (totalRows + 1) / 2;  // ceil(totalRows/2)
        auto rowsThisCore = (coreId == 0) ? rowsCore0 : (totalRows / 2);
        auto cols = tla::get<1>(dstTensor.shape());
        auto gmStride = tla::get<0>(dstTensor.stride());

        auto baseOffset = dstTensor.layout()(dstTensor.coord());
        auto coreOffset = baseOffset + coreId * rowsCore0 * gmStride;

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = rowsThisCore;
        intriParams.blockLen = cols / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = (gmStride - cols) / ELE_NUM_PER_C0;

        // SPLIT_M rounds totalRows up to even, each sub-core gets half
        auto paddedRows = RoundUp(totalRows, 2);
        auto rowsPerCoreInUB = paddedRows / 2;
        auto elementsPerCore = rowsPerCoreInUB * cols;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(transEventId);

        ScaleAndCast(elementsPerCore, transInSlot);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(transEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(transEventId);

        AscendC::DataCopy(dstTensor.data()[coreOffset], ubTransOut, intriParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(transEventId);
    }

private:
    CATLASS_DEVICE void ScaleAndCast(uint32_t count, uint32_t transInSlot)
    {
        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));
        auto src = (__ubuf__ ElementAccumulator*)ubTransIn[transInSlot].GetPhyAddr();
        auto dst = (__ubuf__ Element*)ubTransOut.GetPhyAddr();
        AscendC::VF_CALL<ScaleAndCastVf>(src, dst, scale, count, repeatTimes);
    }

    __simd_vf__ static inline void ScaleAndCastVf(__ubuf__ ElementAccumulator* src, __ubuf__ Element* dst,
                                                  ElementAccumulator scale, uint32_t count, uint32_t repeatTimes)
    {
        constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(ElementAccumulator);
        AscendC::MicroAPI::RegTensor<ElementAccumulator> value;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<ElementAccumulator>(count);
            AscendC::MicroAPI::LoadAlign(value, src + i * ONE_REP_ELM);
            AscendC::MicroAPI::Muls(value, value, scale, mask);
            if constexpr (!std::is_same<Element, ElementAccumulator>::value) {
                Catlass::Epilogue::RegBase::CastDownStore<Element, ElementAccumulator>(
                    dst + i * ONE_REP_ELM, value, mask);
            } else {
                AscendC::MicroAPI::StoreAlign(dst + i * ONE_REP_ELM, value, mask);
            }
        }
    }

    Arch::CrossCoreFlag cubeReady;
    uint32_t transEventId;
    ElementAccumulator scale{static_cast<ElementAccumulator>(1)};

    AscendC::LocalTensor<ElementAccumulator> ubTransIn[TileBuffer::TRANS_IN_CNT];
    AscendC::LocalTensor<Element> ubTransOut;
};

}  // namespace Catlass::Epilogue::Block
