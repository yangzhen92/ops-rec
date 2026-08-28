/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_A5_H_
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_A5_H_

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "simt_api/device_sync_functions.h"
#include "device_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/device_warp_functions.h"

namespace Catlass::Epilogue::Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Catlass::Epilogue::Block

namespace Catlass::Epilogue::Block {
constexpr int32_t BLOCK_M = 128;
constexpr int32_t BLOCK_N = 128;

namespace {
constexpr float MIN_VALUE = -3e38f;
}

template <typename ElementS>
__simt_vf__ __aicore__ LAUNCH_BOUND(2048) inline void BuildArbitraryMaskSimt(uint32_t m, uint32_t n,
                                                                             __gm__ int32_t* arbitraryFuncGm,
                                                                             __ubuf__ ElementS* inputUb,
                                                                             int32_t qSeqOffset, uint32_t kvSeqOffset)
{
    constexpr uint32_t WARP_SIZE = 32;
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    uint32_t warpIdx = tid / WARP_SIZE;  // Warp索引
    uint32_t laneIdx = tid % WARP_SIZE;  // Lane索引
    const uint32_t warpsPerBlock = AscendC::Simt::GetThreadNum() / WARP_SIZE;

    // 每个Warp处理一行或多行
    for (uint32_t rowIdx = warpIdx; rowIdx < m; rowIdx += warpsPerBlock) {
        auto gQIdx = qSeqOffset + rowIdx;

        __gm__ int4* afVecPtr = reinterpret_cast<__gm__ int4*>(arbitraryFuncGm + 4 * gQIdx);
        int4 afVec = *afVecPtr;
        int32_t af1 = afVec.x;
        int32_t af2 = afVec.y;
        int32_t af3 = afVec.z;
        int32_t af4 = afVec.w;

        uint32_t colOffsetBase = rowIdx * BLOCK_N;
#pragma unroll
        for (uint32_t colIdx = laneIdx; colIdx < n; colIdx += WARP_SIZE) {
            auto gKvIdx = kvSeqOffset + colIdx;
            uint32_t elemIdx = colOffsetBase + colIdx;
            if (!((gKvIdx >= af1 && gKvIdx < af2) || (gKvIdx >= af3 && gKvIdx < af4))) {
                inputUb[elemIdx] = MIN_VALUE;
            }
        }
    }
}

template <class L1TileShape_, class PType_, class SType_, class MaskType_, bool ATTENTION_MASK_FLAG_>
class BlockEpilogue<EpilogueAscend950FASoftmax<ATTENTION_MASK_FLAG_>, L1TileShape_, PType_, SType_, MaskType_> {
public:
    using DispatchPolicy = EpilogueAscend950FASoftmax<ATTENTION_MASK_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using ElementP = typename PType_::Element;
    using ElementS = typename SType_::Element;
    using ElementMask = typename MaskType_::Element;
    using LayoutTagMask = typename MaskType_::Layout;
    using LayoutTagP = typename PType_::Layout;

    static constexpr uint32_t S1_BASE_SIZE = tla::get<0>(L1TileShape{});
    static constexpr uint32_t S2_BASE_SIZE = tla::get<1>(L1TileShape{});
    static constexpr bool ATTENTION_MASK_FLAG = ATTENTION_MASK_FLAG_;
    static constexpr int32_t HALF_S1_BASE_SIZE = S1_BASE_SIZE >> 1;
    static constexpr int32_t HALF_VEC_SIZE = HALF_S1_BASE_SIZE * sizeof(ElementS);
    static constexpr int32_t HALF_MASK_BLOCK_SIZE = HALF_S1_BASE_SIZE * S2_BASE_SIZE * sizeof(ElementMask);
    static constexpr int32_t HALF_SCM_BLOCK_SIZE = HALF_S1_BASE_SIZE * S2_BASE_SIZE * sizeof(ElementP);
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementP);

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, ElementS scaleValue_, uint32_t& ubBufAddrStart)
    {
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        scaleValue = scaleValue_;
        expSumUb = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_VEC_SIZE;
        nowMaxUb = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_VEC_SIZE;
        for (uint32_t i = 0; i < TASK_NUM2; i++) {
            vf1OutUbList[i] = resource.ubBuf.template GetBufferByByte<ElementP>(ubBufAddrStart);
            ubBufAddrStart += HALF_SCM_BLOCK_SIZE;
            if constexpr (ATTENTION_MASK_FLAG) {
                attenMaskUbList[i] = resource.ubBuf.template GetBufferByByte<ElementMask>(ubBufAddrStart);
                ubBufAddrStart += HALF_MASK_BLOCK_SIZE;
                eventUbMaskVMTE2List[i] = eventVMTE2++;
                eventUbMaskMTE2VList[i] = eventMTE2V++;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbMaskVMTE2List[i]);
            }
        }

        for (uint32_t i = 0; i < TASK_NUM3; i++) {
            eventUbPMTE3VList[i] = eventMTE3V++;
            eventUbPVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbPMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        if constexpr (ATTENTION_MASK_FLAG) {
            for (uint32_t i = 0; i < TASK_NUM2; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbMaskVMTE2List[i]);
            }
        }
        for (uint32_t i = 0; i < TASK_NUM3; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbPMTE3VList[i]);
        }
    }

    template <class TensorDst, class TensorSrc, class TensorMask>
    CATLASS_DEVICE void operator()(TensorDst& vf1OutL1, AscendC::LocalTensor<ElementS>& sumUb,
                                   AscendC::LocalTensor<ElementS>& lastMaxUb, AscendC::LocalTensor<ElementS>& expMaxUb,
                                   TensorSrc& bmm1Res, TensorMask attenMaskGm, bool isUpdate, int8_t taskIdMod2,
                                   int8_t taskIdMod3, uint64_t MM1_RES_INTRA_EVENT, uint64_t SYNC_V1_C2_FLAG)
    {
        uint32_t m = tla::get<0>(bmm1Res.shape());
        uint32_t n = tla::get<1>(bmm1Res.shape());
        uint32_t blockStride = AscendC::CeilDivision(m, ELE_NUM_PER_C0) * ELE_NUM_PER_C0;
        auto vf1UbLayout = tla::MakeLayout(
            tla::MakeShape(tla::MakeShape(m, tla::Int<1>{}),
                           tla::MakeShape(tla::Int<ELE_NUM_PER_C0>{},
                                          AscendC::CeilDivision(S2_BASE_SIZE, tla::Int<ELE_NUM_PER_C0>{}))),
            tla::MakeStride(tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_C0>{} * m),
                            tla::MakeStride(tla::Int<1>{}, blockStride * ELE_NUM_PER_C0)));
        auto vf1OutUb = tla::MakeTensor(vf1OutUbList[taskIdMod2], vf1UbLayout, Arch::PositionUB{});

        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementS));
        uint32_t tailN = (n - 1) % vlSize + 1;
        uint32_t tailM = m;

        __ubuf__ ElementP* outputAddr = (__ubuf__ ElementP*)vf1OutUb.data().GetPhyAddr();
        __ubuf__ ElementS* inputAddr = (__ubuf__ ElementS*)bmm1Res.data().GetPhyAddr();
        __ubuf__ ElementMask* maskUbAddr;
        __ubuf__ ElementMask* maskUbUnrollAddr;
        __ubuf__ ElementS* lastMaxUbAddr;
        __ubuf__ ElementS* nowMaxUbAddr;
        __ubuf__ ElementS* expSumUbAddr;
        if constexpr (ATTENTION_MASK_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(taskIdMod2);
            auto layoutUb = tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(S2_BASE_SIZE, tla::Int<1>{}));
            auto attenMaskUb = tla::MakeTensor(attenMaskUbList[taskIdMod2], layoutUb, Arch::PositionUB{});
            using CopyGmToUbMask = Tile::CopyGm2UbTla<ArchTag, TensorMask, decltype(attenMaskUb)>;
            CopyGmToUbMask copyGm2UbMask;
            copyGm2UbMask(attenMaskUb, attenMaskGm);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(taskIdMod2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(taskIdMod2);
            maskUbAddr = (__ubuf__ ElementMask*)attenMaskUb.data().GetPhyAddr();
            maskUbUnrollAddr = (__ubuf__ ElementMask*)(attenMaskUb.data().GetPhyAddr() + FLOAT_REP_SIZE);
        }

        if (likely(isUpdate)) {
            lastMaxUbAddr = (__ubuf__ ElementS*)lastMaxUb.GetPhyAddr();
            nowMaxUbAddr = (__ubuf__ ElementS*)nowMaxUb.GetPhyAddr();
            expSumUbAddr = (__ubuf__ ElementS*)expSumUb.GetPhyAddr();
        } else {
            nowMaxUbAddr = (__ubuf__ ElementS*)lastMaxUb.GetPhyAddr();
            expSumUbAddr = (__ubuf__ ElementS*)sumUb.GetPhyAddr();
        }

        if (n == 128) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N128, ATTENTION_MASK_FLAG>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue);
        } else if (n <= 64) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N0_64, ATTENTION_MASK_FLAG>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue);
        } else {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N65_127, ATTENTION_MASK_FLAG>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue);
        }

        if constexpr (ATTENTION_MASK_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(taskIdMod2);
        }

        if (likely(isUpdate)) {
            UpdateMax<ElementS>(nowMaxUbAddr, lastMaxUbAddr, tailM);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        if (unlikely(n > 64)) {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N128>(outputAddr, inputAddr, nowMaxUbAddr,
                                                                                  expSumUbAddr, m, blockStride);
        } else {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(outputAddr, inputAddr, nowMaxUbAddr,
                                                                                   expSumUbAddr, m, blockStride);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT);

        if (likely(m != 0)) {
            using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
            CopyUbToL1P copyUb2L1P;
            copyUb2L1P(vf1OutL1, vf1OutUb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_C2_FLAG);

        if (likely(isUpdate)) {
            __ubuf__ ElementS* sumUbAddr = (__ubuf__ ElementS*)sumUb.GetPhyAddr();
            __ubuf__ ElementS* expMaxUbAddr = (__ubuf__ ElementS*)expMaxUb.GetPhyAddr();
            UpdateExpSumAndExpMax<ElementS>(sumUbAddr, expMaxUbAddr, lastMaxUbAddr, expSumUbAddr, nowMaxUbAddr, tailM);
        }
    }

    template <class TensorDst, class TensorSrc, class TensorMask, class TensorArbitraryFunc>
    CATLASS_DEVICE void operator()(
        TensorDst& vf1OutL1, AscendC::LocalTensor<ElementS>& sumUb, AscendC::LocalTensor<ElementS>& lastMaxUb,
        AscendC::LocalTensor<ElementS>& expMaxUb, TensorSrc& bmm1Res, TensorMask attenMaskGm,
        TensorArbitraryFunc arbitraryFuncGm, bool isUpdate, int8_t taskIdMod2, int8_t taskIdMod3,
        uint64_t MM1_RES_INTRA_EVENT, uint64_t SYNC_V1_C2_FLAG,
        int32_t blockType,  // 0: full block, 1: mask block, 2: empty block, mask block需要构建mask
        int32_t qSeqIdx, uint32_t kvSeqIdx)
    {
        uint32_t m = tla::get<0>(bmm1Res.shape());
        uint32_t n = tla::get<1>(bmm1Res.shape());
        uint32_t blockStride = AscendC::CeilDivision(m, ELE_NUM_PER_C0) * ELE_NUM_PER_C0;
        auto vf1UbLayout = tla::MakeLayout(
            tla::MakeShape(tla::MakeShape(m, tla::Int<1>{}),
                           tla::MakeShape(tla::Int<ELE_NUM_PER_C0>{},
                                          AscendC::CeilDivision(S2_BASE_SIZE, tla::Int<ELE_NUM_PER_C0>{}))),
            tla::MakeStride(tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_C0>{} * m),
                            tla::MakeStride(tla::Int<1>{}, blockStride * ELE_NUM_PER_C0)));
        auto vf1OutUb = tla::MakeTensor(vf1OutUbList[taskIdMod2], vf1UbLayout, Arch::PositionUB{});

        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementS));
        uint32_t tailN = (n - 1) % vlSize + 1;
        uint32_t tailM = m;

        __ubuf__ ElementP* outputAddr = (__ubuf__ ElementP*)vf1OutUb.data().GetPhyAddr();
        __ubuf__ ElementS* inputAddr = (__ubuf__ ElementS*)bmm1Res.data().GetPhyAddr();
        __ubuf__ ElementMask* maskUbAddr;
        __ubuf__ ElementMask* maskUbUnrollAddr;
        __ubuf__ ElementS* lastMaxUbAddr;
        __ubuf__ ElementS* nowMaxUbAddr;
        __ubuf__ ElementS* expSumUbAddr;
        if (ATTENTION_MASK_FLAG && blockType == 1) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(taskIdMod2);
            auto layoutUb = tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(S2_BASE_SIZE, tla::Int<1>{}));
            auto attenMaskUb = tla::MakeTensor(attenMaskUbList[taskIdMod2], layoutUb, Arch::PositionUB{});

            __gm__ int32_t* arbitraryFuncGmPtr = (__gm__ int32_t*)arbitraryFuncGm.data().GetPhyAddr();
            __ubuf__ ElementMask* maskUbPtr = (__ubuf__ ElementMask*)attenMaskUb.data().GetPhyAddr();

            int32_t qSeqOffset = AscendC::GetSubBlockIdx() * m + BLOCK_M * qSeqIdx;
            int32_t kvSeqOffset = kvSeqIdx * BLOCK_N;

            AscendC::Simt::VF_CALL<BuildArbitraryMaskSimt<ElementS>>(
                AscendC::Simt::Dim3(2048), m, n, arbitraryFuncGmPtr, inputAddr, qSeqOffset, kvSeqOffset);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(taskIdMod2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(taskIdMod2);
            maskUbAddr = (__ubuf__ ElementMask*)attenMaskUb.data().GetPhyAddr();
            maskUbUnrollAddr = (__ubuf__ ElementMask*)(attenMaskUb.data().GetPhyAddr() + FLOAT_REP_SIZE);
        }

        if (likely(isUpdate)) {
            lastMaxUbAddr = (__ubuf__ ElementS*)lastMaxUb.GetPhyAddr();
            nowMaxUbAddr = (__ubuf__ ElementS*)nowMaxUb.GetPhyAddr();
            expSumUbAddr = (__ubuf__ ElementS*)expSumUb.GetPhyAddr();
        } else {
            nowMaxUbAddr = (__ubuf__ ElementS*)lastMaxUb.GetPhyAddr();
            expSumUbAddr = (__ubuf__ ElementS*)sumUb.GetPhyAddr();
        }
        if (n == 128) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N128, false>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue, blockType);
        } else if (n <= 64) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N0_64, false>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue, blockType);
        } else {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N65_127, false>(
                inputAddr, maskUbAddr, maskUbUnrollAddr, nowMaxUbAddr, m, tailN, scaleValue, blockType);
        }

        if (ATTENTION_MASK_FLAG && blockType == 1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(taskIdMod2);
        }

        if (likely(isUpdate)) {
            UpdateMax<ElementS>(nowMaxUbAddr, lastMaxUbAddr, tailM);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        if (unlikely(n > 64)) {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N128>(outputAddr, inputAddr, nowMaxUbAddr,
                                                                                  expSumUbAddr, m, blockStride);
        } else {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(outputAddr, inputAddr, nowMaxUbAddr,
                                                                                   expSumUbAddr, m, blockStride);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT);

        if (likely(m != 0)) {
            using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(vf1OutUb), TensorDst>;
            CopyUbToL1P copyUb2L1P;
            copyUb2L1P(vf1OutL1, vf1OutUb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_C2_FLAG);

        if (likely(isUpdate)) {
            __ubuf__ ElementS* sumUbAddr = (__ubuf__ ElementS*)sumUb.GetPhyAddr();
            __ubuf__ ElementS* expMaxUbAddr = (__ubuf__ ElementS*)expMaxUb.GetPhyAddr();
            UpdateExpSumAndExpMax<ElementS>(sumUbAddr, expMaxUbAddr, lastMaxUbAddr, expSumUbAddr, nowMaxUbAddr, tailM);
        }
    }

private:
    static constexpr uint32_t TASK_NUM2 = 2;
    static constexpr uint32_t TASK_NUM3 = 3;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr int64_t BLOCK_BYTES = 32;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint16_t FLOAT_REP_SIZE = 64;
    static constexpr uint16_t DOUBLE_FLOAT_REP_SIZE = 128;

    int32_t eventUbMaskVMTE2List[TASK_NUM2];
    int32_t eventUbMaskMTE2VList[TASK_NUM2];
    int32_t eventUbPMTE3VList[TASK_NUM3];
    int32_t eventUbPVMTE3List[TASK_NUM3];

    ElementS scaleValue;
    AscendC::LocalTensor<uint8_t> attenMaskUbList[TASK_NUM2];
    AscendC::LocalTensor<ElementP> vf1OutUbList[TASK_NUM2];
    AscendC::LocalTensor<ElementS> expSumUb;
    AscendC::LocalTensor<ElementS> nowMaxUb;

    enum class NRangeIndex {
        N0_64 = 0,
        N65_127,
        N128,
        N128_INF
    };

    template <typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange, bool HasAtten = false>
    __simd_vf__ inline void ComputeMaskandScale(__ubuf__ ElementS* srcUb, __ubuf__ uint8_t* maskUb,
                                                __ubuf__ uint8_t* maskUbUnroll, __ubuf__ ElementS* newMaxUb, uint16_t m,
                                                uint32_t tailN, ElementS dScale)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> maxVreg;
        RegTensor<float> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare0;
        MaskReg pregCompare1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<uint32_t>(tailN);

        if constexpr (NRange < NRangeIndex::N128 || HasAtten) {
            Duplicate(minVreg, MIN_VALUE);
        }
        for (uint16_t i = 0; i < m; ++i) {
            if constexpr (NRange > NRangeIndex::N0_64) {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                LoadAlign(srcVreg1, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
                Muls(srcVreg0, srcVreg0, dScale, pregFull);
                Muls(srcVreg1, srcVreg1, dScale, pregTailN);
                if constexpr (HasAtten) {
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare0, (__ubuf__ uint32_t*&)maskUb, S2BaseSize);
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare1, (__ubuf__ uint32_t*&)maskUbUnroll, S2BaseSize);
                    Select(srcVreg0, minVreg, srcVreg0, pregCompare0);
                    Select(srcVreg1, minVreg, srcVreg1, pregCompare1);
                }
                if constexpr (NRange < NRangeIndex::N128) {
                    Select(srcVreg1, srcVreg1, minVreg, pregTailN);
                }
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg1,
                                                               pregFull);
                Max(maxTmpVreg, srcVreg0, srcVreg1, pregFull);
                ReduceMax(maxVreg, maxTmpVreg, pregFull);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                Muls(srcVreg0, srcVreg0, dScale, pregTailN);
                if constexpr (HasAtten) {
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare0, (__ubuf__ uint32_t*&)maskUb, S2BaseSize);
                    Select(srcVreg0, minVreg, srcVreg0, pregCompare0);
                }
                Select(srcVreg0, srcVreg0, minVreg, pregTailN);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                ReduceMax(maxVreg, srcVreg0, pregFull);
            }
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxVreg, maxUreg, 1);
        }
        StoreUnAlignPost(newMaxUb, maxUreg, 0);
    }

    template <typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange, bool HasAtten = false>
    __simd_vf__ inline void ComputeMaskandScale(__ubuf__ ElementS* srcUb, __ubuf__ uint8_t* maskUb,
                                                __ubuf__ uint8_t* maskUbUnroll, __ubuf__ ElementS* newMaxUb, uint16_t m,
                                                uint32_t tailN, ElementS dScale, int32_t blockType)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> maxVreg;
        RegTensor<float> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare0;
        MaskReg pregCompare1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<uint32_t>(tailN);

        if (NRange < NRangeIndex::N128 || (HasAtten && blockType == 1)) {
            Duplicate(minVreg, MIN_VALUE);
        }
        for (uint16_t i = 0; i < m; ++i) {
            if constexpr (NRange > NRangeIndex::N0_64) {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                LoadAlign(srcVreg1, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
                Muls(srcVreg0, srcVreg0, dScale, pregFull);
                Muls(srcVreg1, srcVreg1, dScale, pregTailN);
                if (HasAtten && blockType == 1) {
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare0, (__ubuf__ uint32_t*&)maskUb, S2BaseSize);
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare1, (__ubuf__ uint32_t*&)maskUbUnroll, S2BaseSize);
                    Select(srcVreg0, minVreg, srcVreg0, pregCompare0);
                    Select(srcVreg1, minVreg, srcVreg1, pregCompare1);
                }
                if constexpr (NRange < NRangeIndex::N128) {
                    Select(srcVreg1, srcVreg1, minVreg, pregTailN);
                }
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg1,
                                                               pregFull);
                Max(maxTmpVreg, srcVreg0, srcVreg1, pregFull);
                ReduceMax(maxVreg, maxTmpVreg, pregFull);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                Muls(srcVreg0, srcVreg0, dScale, pregTailN);
                if (HasAtten && blockType == 1) {
                    LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                        pregCompare0, (__ubuf__ uint32_t*&)maskUb, S2BaseSize);
                    Select(srcVreg0, minVreg, srcVreg0, pregCompare0);
                }
                Select(srcVreg0, srcVreg0, minVreg, pregTailN);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                ReduceMax(maxVreg, srcVreg0, pregFull);
            }
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxVreg, maxUreg, 1);
        }
        StoreUnAlignPost(newMaxUb, maxUreg, 0);
    }

    template <typename ElementS>
    __simd_vf__ inline void UpdateMax(__ubuf__ ElementS* nowMaxUb, __ubuf__ ElementS* lastMaxUb, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> nowMaxVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> maxVreg;

        MaskReg pregTailM = UpdateMask<float>(tailM);
        LoadAlign(lastMaxVreg, lastMaxUb);
        LoadAlign(nowMaxVreg, nowMaxUb);
        Max(maxVreg, nowMaxVreg, lastMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(nowMaxUb, maxVreg, pregTailM);
    }

    template <typename ElementP, typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange>
    __simd_vf__ inline void ComputeExpSubSum(__ubuf__ ElementP* expUb, __ubuf__ ElementS* srcUb,
                                             __ubuf__ ElementS* nowMaxUb, __ubuf__ ElementS* expSumUb, uint16_t m,
                                             uint32_t blockStride)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> expVreg;
        RegTensor<float> expVreg0;
        RegTensor<float> expVreg1;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<ElementP> expDstVreg0;
        RegTensor<ElementP> expDstVreg1;
        RegTensor<ElementP> expDstVreg;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFull16 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementS, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            if constexpr (NRange > NRangeIndex::N0_64) {
                LoadAlign<ElementS, LoadDist::DIST_DINTLV_B32>(srcVreg0, srcVreg1, srcUb + i * S2BaseSize);
                ExpSub(expVreg0, srcVreg0, maxVreg, pregFull);
                ExpSub(expVreg1, srcVreg1, maxVreg, pregFull);
                Add(expVreg, expVreg0, expVreg1, pregFull);
                Cast<ElementP, ElementS, castTraitZero>(expDstVreg0, expVreg0, pregFull);
                Cast<ElementP, ElementS, castTraitOne>(expDstVreg1, expVreg1, pregFull);
                Or((RegTensor<uint16_t>&)expDstVreg, (RegTensor<uint16_t>&)expDstVreg0,
                   (RegTensor<uint16_t>&)expDstVreg1, pregFull16);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, expDstVreg, blockStride, REPEAT_STRIDE, pregFull16);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                ExpSub(expVreg, srcVreg0, maxVreg, pregFull);
                Cast<ElementP, ElementS, castTraitZero>(expDstVreg, expVreg, pregFull);
                DeInterleave(expDstVreg0, expDstVreg1, expDstVreg, expDstVreg);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, expDstVreg0, blockStride, REPEAT_STRIDE, pregFull16);
            }
            ReduceSum(expSumVreg, expVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        StoreUnAlignPost(expSumUb, expSumUreg, 0);
    }

    template <typename ElementS>
    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ ElementS* sumUb, __ubuf__ ElementS* expMaxUb,
                                                  __ubuf__ ElementS* maxUb, __ubuf__ ElementS* expSumUb,
                                                  __ubuf__ ElementS* nowMaxUb, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> nowMaxVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<float> brcExpSumVreg;
        MaskReg pregTailM = UpdateMask<float>(tailM);
        LoadAlign(lastMaxVreg, maxUb);
        LoadAlign(nowMaxVreg, nowMaxUb);
        ExpSub(expMaxVreg, lastMaxVreg, nowMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(maxUb, nowMaxVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb);
        LoadAlign(brcExpSumVreg, expSumUb);
        MulDstAdd(expMaxVreg, lastExpSumVreg, brcExpSumVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(sumUb, expMaxVreg, pregTailM);
    }
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FA_SOFTMAX_A5