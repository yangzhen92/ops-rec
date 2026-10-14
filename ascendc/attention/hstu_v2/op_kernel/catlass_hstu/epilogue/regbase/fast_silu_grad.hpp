/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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
 • @file fast_silu_grad.hpp

 • @brief SiLU 激活函数梯度计算实现

 • @description 提供 SiLU 激活函数的反向传播梯度计算，用于 HSTU 注意力机制的 backward 过程

 •              计算公式: d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

 */
#pragma once

#include "../../../catlass_hstu/epilogue/regbase/fast_silu_score.hpp"

namespace Catlass::Epilogue::RegBase {

/**
 • @brief SiLU 梯度向量化快速计算函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam GrabType 梯度数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr SiLU 分数的 UBuf 指针

 • @param ubGradPartPtr 输出梯度的 UBuf 指针

 • @param alpha 注意力分数的缩放系数

 • @param count 有效元素数量

 • @param repeatTimes 重复次数 (向量化的循环次数)

 • @description 实现 SiLU 梯度的向量化计算:

 •              1. 重新计算 SiLU score (调用 SiluScore)

 •              2. 计算 sigmoid 的导数: Z = 1 - sigmoid

 •              3. 计算梯度: T = T + S * (1 - Z) = sigmoid * (1 + x * (1 - sigmoid))

 •              4. 乘以 alpha 系数

 •              5. 存储梯度结果

 */
template <typename Type, typename AccType, typename GrabType, bool HAS_RAB, bool NEED_MASK>
__simd_callee__ inline void FastSiluGradVfImpl(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                               __ubuf__ Type* ubMaskPtr, __ubuf__ Type* ubSiluScorePtr,
                                               __ubuf__ GrabType* ubGradPartPtr, AccType alpha, uint32_t count,
                                               uint32_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> vregScore;
    AscendC::MicroAPI::RegTensor<AccType> vregDenom;
    AscendC::MicroAPI::RegTensor<AccType> vregTmp;
    AscendC::MicroAPI::RegTensor<AccType> vregSilu;
    AscendC::MicroAPI::RegTensor<AccType> vregOnes;
    AscendC::MicroAPI::RegTensor<AccType> vregZeros;
    AscendC::MicroAPI::MaskReg maskReg;

    AscendC::MicroAPI::Duplicate(vregOnes, 1.0f);
    AscendC::MicroAPI::Duplicate(vregZeros, 0.0f);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        maskReg = AscendC::MicroAPI::UpdateMask<AccType>(count);

        SiluScoreBackward<Type, AccType, HAS_RAB, NEED_MASK>(
            ubSPtr + i * oneRepElm, ubRabPtr + i * oneRepElm, ubMaskPtr + i * oneRepElm,
            ubSiluScorePtr + i * oneRepElm, vregScore, vregDenom, vregTmp, vregSilu, vregOnes, vregZeros, alpha,
            maskReg);

        AscendC::MicroAPI::Sub(vregTmp, vregOnes, vregDenom, maskReg);
        AscendC::MicroAPI::MulAddDst(vregDenom, vregSilu, vregTmp, maskReg);

        if constexpr (!std::is_same<GrabType, AccType>::value) {
            CastDownStore<GrabType, AccType>(ubGradPartPtr + i * oneRepElm, vregDenom, maskReg);
        } else {
            AscendC::MicroAPI::StoreAlign(ubGradPartPtr + i * oneRepElm, vregDenom, maskReg);
        }
    }
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubMaskPtr,
                                       __ubuf__ Type* ubSiluScorePtr, __ubuf__ GrabType* ubGradPartPtr, AccType alpha,
                                       uint32_t count, uint32_t repeatTimes, bool needMask)
{
    if (needMask) {
        FastSiluGradVfImpl<Type, AccType, GrabType, HAS_RAB, true>(ubSPtr, ubRabPtr, ubMaskPtr, ubSiluScorePtr,
                                                                   ubGradPartPtr, alpha, count, repeatTimes);
    } else {
        FastSiluGradVfImpl<Type, AccType, GrabType, HAS_RAB, false>(ubSPtr, ubRabPtr, ubMaskPtr, ubSiluScorePtr,
                                                                    ubGradPartPtr, alpha, count, repeatTimes);
    }
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradPackedVf(
    __ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubSiluScorePtr,
    __ubuf__ GrabType* ubGradPartPtr, __ubuf__ uint32_t* ubPackedMaskPtr, AccType alpha, uint32_t rows, uint32_t cols,
    uint32_t maskWordsPerRow)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));
    constexpr uint32_t MASK_SLOT_WORDS = 8;
    constexpr uint32_t OUTPUT_REP_ELM = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(Type));
    static_assert(ONE_REP_ELM == 64, "packed 128-bit mask expects two FP32 RegBase repeats");
    static_assert(OUTPUT_REP_ELM == 128, "packed 128-bit mask expects one FP16/BF16 output repeat");

    RegTensor<AccType> vregA;
    RegTensor<AccType> vregZ;
    RegTensor<AccType> vregT;
    RegTensor<AccType> vregS;
    RegTensor<AccType> vregOnes;
    RegTensor<AccType> vregZeros;
    MaskReg sourceMaskTensor;
    MaskReg fullMask = CreateMask<AccType, MaskPattern::ALL>();
    Duplicate(vregOnes, static_cast<AccType>(1));
    Duplicate(vregZeros, static_cast<AccType>(0));

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = row * cols;
        const uint32_t repeatTimes = (cols + ONE_REP_ELM - 1) / ONE_REP_ELM;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            const uint32_t valueOffset = rowOffset + repeat * ONE_REP_ELM;
            uint32_t valid = cols - repeat * ONE_REP_ELM;
            if (valid > ONE_REP_ELM) valid = ONE_REP_ELM;
            fullMask = UpdateMask<AccType>(valid);
            LoadAlign(vregA, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> vregRab;
                if constexpr (!std::is_same<Type, AccType>::value) {
                    CastUpLoad<AccType, Type>(vregRab, ubRabPtr + valueOffset, fullMask);
                } else {
                    LoadAlign(vregRab, ubRabPtr + valueOffset);
                }
                Axpy(vregA, vregRab, alpha, fullMask);
            }
            ExpSub(vregZ, vregZeros, vregA, fullMask);
            Add(vregZ, vregOnes, vregZ, fullMask);
            Div(vregZ, vregOnes, vregZ, fullMask);
            LoadAlign<uint32_t, MaskDist::DIST_US>(
                sourceMaskTensor, ubPackedMaskPtr + row * maskWordsPerRow + repeat * 8);
            MaskUnPack(sourceMaskTensor, sourceMaskTensor);
            Mul<AccType, MaskMergeMode::ZEROING>(vregS, vregZ, vregA, sourceMaskTensor);
            if constexpr (!std::is_same<Type, AccType>::value) {
                CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, vregS, fullMask);
            } else {
                StoreAlign(ubSiluScorePtr + valueOffset, vregS, fullMask);
            }

            Sub(vregT, vregOnes, vregZ, fullMask);
            MulAddDst<AccType, MaskMergeMode::ZEROING>(vregZ, vregS, vregT, sourceMaskTensor);
            if constexpr (!std::is_same<GrabType, AccType>::value) {
                CastDownStore<GrabType, AccType>(ubGradPartPtr + valueOffset, vregZ, fullMask);
            } else {
                StoreAlign(ubGradPartPtr + valueOffset, vregZ, fullMask);
            }
        }
    }
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradHistoryVf(
    __ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubSiluScorePtr,
    __ubuf__ GrabType* ubGradPartPtr, AccType alpha, uint32_t rows, uint32_t cols, int32_t qBase, int32_t kBase,
    int32_t seqlenQ, int32_t seqlenK)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(AccType);
    RegTensor<AccType> a, z, s, t, ones, zeros;
    RegTensor<int32_t> colsReg;
    MaskReg validMask;
    MaskReg scoreMask;
    Duplicate(ones, static_cast<AccType>(1));
    Duplicate(zeros, static_cast<AccType>(0));

    for (uint16_t row = 0; row < rows; ++row) {
        const int32_t globalQ = qBase + row;
        const int32_t historyEnd = globalQ + seqlenK - seqlenQ;
        for (uint16_t repeat = 0; repeat < (cols + ONE_REP_ELM - 1) / ONE_REP_ELM; ++repeat) {
            const uint32_t valueOffset = row * cols + repeat * ONE_REP_ELM;
            uint32_t valid = cols - repeat * ONE_REP_ELM;
            if (valid > ONE_REP_ELM) valid = ONE_REP_ELM;
            validMask = UpdateMask<AccType>(valid);
            LoadAlign(a, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> rab;
                if constexpr (!std::is_same<Type, AccType>::value) {
                    CastUpLoad<AccType, Type>(rab, ubRabPtr + valueOffset, validMask);
                } else {
                    LoadAlign(rab, ubRabPtr + valueOffset);
                }
                Axpy(a, rab, alpha, validMask);
            }
            ExpSub(z, zeros, a, validMask);
            Add(z, ones, z, validMask);
            Div(z, ones, z, validMask);

            const int32_t colStart = kBase + repeat * ONE_REP_ELM;
            Arange(colsReg, colStart);
            Compares<int32_t, AscendC::CMPMODE::LE>(scoreMask, colsReg, historyEnd, validMask);
            Select(z, z, zeros, scoreMask);
            Mul(s, z, a, validMask);
            if constexpr (!std::is_same<Type, AccType>::value) {
                CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, s, validMask);
            } else {
                StoreAlign(ubSiluScorePtr + valueOffset, s, validMask);
            }
            Sub(t, ones, z, validMask);
            MulAddDst(z, s, t, validMask);
            if constexpr (!std::is_same<GrabType, AccType>::value) {
                CastDownStore<GrabType, AccType>(ubGradPartPtr + valueOffset, z, validMask);
            } else {
                StoreAlign(ubGradPartPtr + valueOffset, z, validMask);
            }
        }
    }
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradLocalVf(
    __ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr, __ubuf__ Type* ubSiluScorePtr,
    __ubuf__ GrabType* ubGradPartPtr, AccType alpha, uint32_t rows, uint32_t cols, int32_t qBase, int32_t kBase,
    int32_t seqlenQ, int32_t seqlenK, int32_t windowSizeLeft, int32_t windowSizeRight,
    int32_t numContext, int32_t numTarget, int32_t targetGroupSize)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(AccType);
    RegTensor<AccType> a, z, s, t, ones, zeros;
    RegTensor<int32_t> colsReg;
    MaskReg validMask, lowerMask, upperMask, scoreMask, contextMask, outsideTargetMask, targetLowerMask, targetUpperMask;
    Duplicate(ones, static_cast<AccType>(1));
    Duplicate(zeros, static_cast<AccType>(0));
    const int32_t delta = seqlenK - seqlenQ;
    for (uint16_t row = 0; row < rows; ++row) {
        const int32_t center = qBase + row + delta;
        const int32_t lower = windowSizeLeft < 0 ? 0 : center - windowSizeLeft;
        const int32_t upper = windowSizeRight < 0 ? seqlenK - 1 : center + windowSizeRight;
        const int32_t globalQ = qBase + row;
        const bool contextRow = numContext > 0 && globalQ < numContext;
        const bool targetRow = numTarget > 0 && targetGroupSize > 0 && globalQ >= seqlenQ - numTarget;
        for (uint16_t repeat = 0; repeat < (cols + ONE_REP_ELM - 1) / ONE_REP_ELM; ++repeat) {
            const uint32_t valueOffset = row * cols + repeat * ONE_REP_ELM;
            uint32_t valid = cols - repeat * ONE_REP_ELM;
            if (valid > ONE_REP_ELM) valid = ONE_REP_ELM;
            validMask = UpdateMask<AccType>(valid);
            LoadAlign(a, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> rab;
                if constexpr (!std::is_same<Type, AccType>::value) CastUpLoad<AccType, Type>(rab, ubRabPtr + valueOffset, validMask);
                else LoadAlign(rab, ubRabPtr + valueOffset);
                Axpy(a, rab, alpha, validMask);
            }
            ExpSub(z, zeros, a, validMask);
            Add(z, ones, z, validMask);
            Div(z, ones, z, validMask);
            Arange(colsReg, kBase + repeat * ONE_REP_ELM);
            Compares<int32_t, AscendC::CMPMODE::GE>(lowerMask, colsReg, lower, validMask);
            Compares<int32_t, AscendC::CMPMODE::LE>(upperMask, colsReg, upper, validMask);
            And(scoreMask, lowerMask, upperMask, validMask);
            if (contextRow) {
                Compares<int32_t, AscendC::CMPMODE::LT>(contextMask, colsReg, seqlenK - numTarget, validMask);
                Or(scoreMask, scoreMask, contextMask, validMask);
            }
            if (targetRow) {
                const int32_t targetBase = seqlenK - numTarget;
                const int32_t targetRowIndex = globalQ - (seqlenQ - numTarget);
                const int32_t groupBegin = targetBase + targetRowIndex / targetGroupSize * targetGroupSize;
                const int32_t groupEnd = targetBase + targetRowIndex;
                Compares<int32_t, AscendC::CMPMODE::LT>(outsideTargetMask, colsReg, targetBase, validMask);
                And(scoreMask, scoreMask, outsideTargetMask, validMask);
                Compares<int32_t, AscendC::CMPMODE::GE>(targetLowerMask, colsReg, groupBegin, validMask);
                Compares<int32_t, AscendC::CMPMODE::LE>(targetUpperMask, colsReg, groupEnd, validMask);
                And(targetLowerMask, targetLowerMask, targetUpperMask, validMask);
                Or(scoreMask, scoreMask, targetLowerMask, validMask);
            }
            Select(z, z, zeros, scoreMask);
            Mul(s, z, a, validMask);
            if constexpr (!std::is_same<Type, AccType>::value) CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, s, validMask);
            else StoreAlign(ubSiluScorePtr + valueOffset, s, validMask);
            Sub(t, ones, z, validMask);
            MulAddDst(z, s, t, validMask);
            if constexpr (!std::is_same<GrabType, AccType>::value) CastDownStore<GrabType, AccType>(ubGradPartPtr + valueOffset, z, validMask);
            else StoreAlign(ubGradPartPtr + valueOffset, z, validMask);
        }
    }
}

}  // namespace Catlass::Epilogue::RegBase
