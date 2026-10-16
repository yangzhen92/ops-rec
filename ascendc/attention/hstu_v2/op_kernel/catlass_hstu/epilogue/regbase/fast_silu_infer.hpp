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
 • @file fast_silu_infer.hpp

 • @brief SiLU (Swish) 激活函数得分计算实现

 • @description 提供基于寄存器的 SiLU 激活函数前向计算、掩码乘法、RAB 相加等操作，

 •              用于 HSTU 注意力机制中 score = silu(QK^T + RAB) * scale * mask 的计算

 */
#pragma once

#include "../../../catlass_hstu/epilogue/regbase/common.h"

namespace Catlass::Epilogue::RegBase {

/**
 • @brief 加上相对位置注意力偏置 (RAB)

 • @tparam Type 输入数据类型

 • @tparam AccType 累加器数据类型

 • @tparam RegType 寄存器类型

 • @param dst 目标寄存器，存储计算结果

 • @param src 源寄存器，输入的注意力分数

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param maskReg 掩码寄存器

 • @description 将 RAB (Relative Attention Bias) 加到注意力分数上: dst = src + RAB

 */
template <typename Type, typename AccType, typename RegType>
__simd_callee__ inline void AddRab(RegType& dst, __ubuf__ Type* ubRabPtr, AccType alpha,
                                   AscendC::MicroAPI::MaskReg& maskReg)
{
    RegType vregRab;

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastUpLoad<AccType, Type>(vregRab, ubRabPtr, maskReg);
    } else {
        AscendC::MicroAPI::LoadAlign(vregRab, ubRabPtr);
    }

    AscendC::MicroAPI::Axpy(dst, vregRab, alpha, maskReg);  // A = A + Rab * alpha
}

/**
 • @brief 掩码乘法操作

 • @tparam Type 输入数据类型

 • @tparam AccType 累加器数据类型

 • @tparam RegType 寄存器类型

 • @param dst 目标寄存器，存储计算结果

 • @param src 源寄存器，输入的注意力分数

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param maskReg 掩码寄存器

 • @description 将掩码应用到注意力分数上: dst = src * mask，用于屏蔽无效位置

 */
template <typename Type, typename AccType, typename RegType>
__simd_callee__ inline void MulMask(RegType& dst, RegType& src, __ubuf__ Type* ubMaskPtr,
                                    AscendC::MicroAPI::MaskReg& maskReg)
{
    RegType vregMask;

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastUpLoad<AccType, Type>(vregMask, ubMaskPtr, maskReg);
    } else {
        AscendC::MicroAPI::LoadAlign(vregMask, ubMaskPtr);
    }
    AscendC::MicroAPI::Mul(dst, src, vregMask, maskReg);  // Z = Z * mask
}

/**
 • @brief SiLU Score 计算核心函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @tparam HAS_MASK 是否有掩码

 • @tparam RegType 寄存器类型

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr 输出 SiLU 分数的 UBuf 指针

 • @param vregA 工作寄存器 A

 • @param vregZ 工作寄存器 Z (sigmoid 结果)

 • @param vregT 工作寄存器 T (中间值)

 • @param vregS 工作寄存器 S (最终结果)

 • @param vregOnes 常量寄存器，值为 1.0

 • @param vregAlpha Alpha 系数寄存器

 • @param vregScale Scale 系数寄存器

 • @param maskReg 掩码寄存器

 • @description 实现完整的 SiLU Score 计算流程:

 •              1. 加载 QK^T 分数

 •              2. 加上 RAB (如果 HAS_RAB)

 •              3. 乘以 alpha

 •              4. 计算 sigmoid

 •              5. 乘以掩码 (如果 HAS_MASK)

 •              6. 乘以 scale

 •              7. 计算 silu(x) = x * sigmoid(x) 并存储

 */
template <typename Type, typename AccType, bool HAS_RAB, bool HAS_MASK, typename RegType>
__simd_callee__ inline void SiluScoreForward(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                             __ubuf__ Type* ubMaskPtr, __ubuf__ Type* ubSiluScorePtr,
                                             RegType& vregScore, RegType& vregDenom, RegType& vregSilu,
                                             RegType& vregOnes, RegType& vregZeros, AccType alpha,
                                             AscendC::MicroAPI::MaskReg& maskReg)
{
    AscendC::MicroAPI::LoadAlign(vregScore, ubSPtr);

    if constexpr (HAS_RAB) {
        RegType vregRab;
        if constexpr (!std::is_same<Type, AccType>::value) {
            CastUpLoad<AccType, Type>(vregRab, ubRabPtr, maskReg);
        } else {
            AscendC::MicroAPI::LoadAlign(vregRab, ubRabPtr);
        }
        AscendC::MicroAPI::Add(vregScore, vregScore, vregRab, maskReg);
    }
    AscendC::MicroAPI::Muls(vregScore, vregScore, alpha, maskReg);
    SiluCore(vregScore, vregDenom, vregSilu, vregOnes, vregZeros, maskReg);

    if constexpr (!std::is_same<Type, AccType>::value) {
        CastDownStore<Type, AccType>(ubSiluScorePtr, vregSilu, maskReg);
    } else {
        AscendC::MicroAPI::StoreAlign(ubSiluScorePtr, vregSilu, maskReg);
    }
}

/**
 • @brief SiLU Score 向量化快速计算函数

 • @tparam Type 输入/输出数据类型

 • @tparam AccType 累加器数据类型

 • @tparam HAS_RAB 是否有相对位置偏置

 • @tparam HAS_MASK 是否有掩码

 • @param ubSPtr SiLU 输入 (QK^T) 的 UBuf 指针

 • @param ubRabPtr RAB 数据的 UBuf 指针

 • @param ubMaskPtr 掩码数据的 UBuf 指针

 • @param ubSiluScorePtr 输出 SiLU 分数的 UBuf 指针

 • @param alpha 注意力分数的缩放系数

 • @param scale SiLU 输出缩放因子

 • @param count 有效元素数量

 • @param repeatTimes 重复次数 (向量化的循环次数)

 • @description 向量化版本的 SiLU Score 计算，通过循环处理多次向量运算，

 •              每次处理一个向量长度 (GetVecLen/sizeof(AccType)) 的数据

 */
template <typename Type, typename AccType, bool HAS_RAB, bool HAS_MASK>
__simd_vf__ inline void FastSiluScoreForwardVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                               __ubuf__ Type* ubMaskPtr, __ubuf__ Type* ubSiluScorePtr, AccType alpha,
                                               AccType scale, uint32_t count, uint32_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> vregScore;
    AscendC::MicroAPI::RegTensor<AccType> vregDenom;
    AscendC::MicroAPI::RegTensor<AccType> vregSilu;
    AscendC::MicroAPI::RegTensor<AccType> vregOnes;
    AscendC::MicroAPI::RegTensor<AccType> vregZeros;
    AscendC::MicroAPI::MaskReg maskReg;

    AscendC::MicroAPI::Duplicate(vregOnes, 1.0f);
    AscendC::MicroAPI::Duplicate(vregZeros, 0.0f);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        // maskReg 必须在每次迭代前重算：Sigmoid 内 Div 等指令会消费/改写 mask 寄存器，
        // 循环外只算一次会导致后续迭代 mask 错误（3dba5595 精度回归根因）
        maskReg = AscendC::MicroAPI::UpdateMask<AccType>(count);
        SiluScoreForward<Type, AccType, HAS_RAB, HAS_MASK>(
            ubSPtr + i * oneRepElm, ubRabPtr + i * oneRepElm, ubMaskPtr + i * oneRepElm, ubSiluScorePtr + i * oneRepElm,
            vregScore, vregDenom, vregSilu, vregOnes, vregZeros, alpha, maskReg);
    }
}

template <typename Type, typename AccType, bool HAS_RAB>
__simd_vf__ inline void FastSiluScorePackedVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                              __ubuf__ Type* ubSiluScorePtr, __ubuf__ uint32_t* ubPackedMaskPtr,
                                              AccType alpha, AccType scale, uint32_t rows, uint32_t cols,
                                              uint32_t maskWordsPerRow)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(AccType);
    RegTensor<AccType> score, denom, silu, ones, zeros, alphaReg;
    RegTensor<Type> outputZeros;
    MaskReg sourceMask;
    MaskReg fullMask = CreateMask<AccType, MaskPattern::ALL>();
    MaskReg outputMask = CreateMask<Type, MaskPattern::ALL>();
    Duplicate(ones, static_cast<AccType>(1));
    Duplicate(zeros, static_cast<AccType>(0));
    Duplicate(alphaReg, alpha);
    Duplicate(outputZeros, static_cast<Type>(0));
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = row * cols;
        const uint32_t repeatTimes = (cols + ONE_REP_ELM - 1) / ONE_REP_ELM;
        StoreAlign(ubSiluScorePtr + rowOffset, outputZeros, outputMask);
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            const uint32_t valueOffset = rowOffset + repeat * ONE_REP_ELM;
            LoadAlign(score, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> rab;
                fullMask = CreateMask<AccType, MaskPattern::ALL>();
                if constexpr (!std::is_same<Type, AccType>::value) {
                    CastUpLoad<AccType, Type>(rab, ubRabPtr + valueOffset, fullMask);
                } else {
                    LoadAlign(rab, ubRabPtr + valueOffset);
                }
                fullMask = CreateMask<AccType, MaskPattern::ALL>();
                Add(score, score, rab, fullMask);
            }
            fullMask = CreateMask<AccType, MaskPattern::ALL>();
            Mul(score, score, alphaReg, fullMask);
            fullMask = CreateMask<AccType, MaskPattern::ALL>();
            SiluCore(score, denom, silu, ones, zeros, fullMask);
            LoadAlign<uint32_t, MaskDist::DIST_US>(sourceMask, ubPackedMaskPtr + row * maskWordsPerRow + repeat * 8);
            MaskUnPack(sourceMask, sourceMask);
            Select(silu, silu, zeros, sourceMask);
            uint32_t valid = cols - repeat * ONE_REP_ELM;
            if (valid > ONE_REP_ELM)
                valid = ONE_REP_ELM;
            fullMask = UpdateMask<AccType>(valid);
            if constexpr (!std::is_same<Type, AccType>::value) {
                CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, silu, fullMask);
            } else {
                StoreAlign(ubSiluScorePtr + valueOffset, silu, fullMask);
            }
        }
    }
}

template <typename Type, typename AccType, bool HAS_RAB>
__simd_vf__ inline void FastSiluScoreCausalVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                              __ubuf__ Type* ubSiluScorePtr, AccType alpha, AccType scale,
                                              uint32_t rows, uint32_t cols, int32_t qBase, int32_t kBase,
                                              int32_t seqlenQ, int32_t seqlenK, int32_t numContext, int32_t numTarget,
                                              int32_t targetGroupSize)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(AccType);
    RegTensor<AccType> a, z, s, ones, zeros, alphaReg;
    RegTensor<int32_t> colsReg;
    MaskReg validMask;
    MaskReg scoreMask;
    Duplicate(ones, static_cast<AccType>(1));
    Duplicate(zeros, static_cast<AccType>(0));
    Duplicate(alphaReg, alpha);

    for (uint16_t row = 0; row < rows; ++row) {
        const int32_t globalQ = qBase + row;
        const int32_t historyEnd = globalQ + seqlenK - seqlenQ;
        const bool contextRow = numContext > 0 && globalQ < numContext;
        const bool targetRow = numTarget > 0 && targetGroupSize > 0 && globalQ >= seqlenQ - numTarget;
        for (uint16_t repeat = 0; repeat < (cols + ONE_REP_ELM - 1) / ONE_REP_ELM; ++repeat) {
            const uint32_t valueOffset = row * cols + repeat * ONE_REP_ELM;
            uint32_t valid = cols - repeat * ONE_REP_ELM;
            if (valid > ONE_REP_ELM)
                valid = ONE_REP_ELM;
            validMask = UpdateMask<AccType>(valid);
            LoadAlign(a, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> rab;
                if constexpr (!std::is_same<Type, AccType>::value) {
                    CastUpLoad<AccType, Type>(rab, ubRabPtr + valueOffset, validMask);
                } else {
                    LoadAlign(rab, ubRabPtr + valueOffset);
                }
                Add(a, a, rab, validMask);
            }
            Mul(a, a, alphaReg, validMask);
            ExpSub(z, zeros, a, validMask);
            Add(z, ones, z, validMask);
            Div(z, ones, z, validMask);
            Mul(s, z, a, validMask);

            const int32_t colStart = kBase + repeat * ONE_REP_ELM;
            Arange(colsReg, colStart);
            Compares<int32_t, AscendC::CMPMODE::LE>(scoreMask, colsReg, historyEnd, validMask);
            if (contextRow) {
                MaskReg contextMask;
                const int32_t historyWidth = seqlenK - numTarget;
                Compares<int32_t, AscendC::CMPMODE::LT>(contextMask, colsReg, historyWidth, validMask);
                Or(scoreMask, scoreMask, contextMask, validMask);
            }
            if (targetRow) {
                const int32_t targetBase = seqlenK - numTarget;
                const int32_t group = (globalQ - (seqlenQ - numTarget)) / targetGroupSize;
                const int32_t groupBegin = targetBase + group * targetGroupSize;
                const int32_t groupEnd = groupBegin + targetGroupSize - 1;
                MaskReg historyRegion;
                MaskReg groupMask;
                MaskReg upperMask;
                Compares<int32_t, AscendC::CMPMODE::LT>(historyRegion, colsReg, targetBase, validMask);
                And(historyRegion, historyRegion, scoreMask, validMask);
                Compares<int32_t, AscendC::CMPMODE::GE>(groupMask, colsReg, groupBegin, validMask);
                Compares<int32_t, AscendC::CMPMODE::LE>(upperMask, colsReg,
                                                        groupEnd < historyEnd ? groupEnd : historyEnd, validMask);
                And(groupMask, groupMask, upperMask, validMask);
                Or(scoreMask, historyRegion, groupMask, validMask);
            }
            Select(s, s, zeros, scoreMask);
            if constexpr (!std::is_same<Type, AccType>::value) {
                CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, s, validMask);
            } else {
                StoreAlign(ubSiluScorePtr + valueOffset, s, validMask);
            }
        }
    }
}

template <typename Type, typename AccType, bool HAS_RAB>
__simd_vf__ inline void FastSiluScoreLocalVf(__ubuf__ AccType* ubSPtr, __ubuf__ Type* ubRabPtr,
                                             __ubuf__ Type* ubSiluScorePtr, AccType alpha, AccType scale, uint32_t rows,
                                             uint32_t cols, int32_t qBase, int32_t kBase, int32_t seqlenQ,
                                             int32_t seqlenK, int32_t windowSizeLeft, int32_t windowSizeRight,
                                             int32_t numContext, int32_t numTarget, int32_t targetGroupSize)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ONE_REP_ELM = AscendC::GetVecLen() / sizeof(AccType);
    RegTensor<AccType> a, z, s, ones, zeros, alphaReg;
    RegTensor<int32_t> colsReg;
    MaskReg validMask, lowerMask, upperMask, scoreMask, contextMask, outsideTargetMask, targetLowerMask,
        targetUpperMask;
    Duplicate(ones, static_cast<AccType>(1));
    Duplicate(zeros, static_cast<AccType>(0));
    Duplicate(alphaReg, alpha);
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
            if (valid > ONE_REP_ELM)
                valid = ONE_REP_ELM;
            validMask = UpdateMask<AccType>(valid);
            LoadAlign(a, ubSPtr + valueOffset);
            if constexpr (HAS_RAB) {
                RegTensor<AccType> rab;
                if constexpr (!std::is_same<Type, AccType>::value)
                    CastUpLoad<AccType, Type>(rab, ubRabPtr + valueOffset, validMask);
                else
                    LoadAlign(rab, ubRabPtr + valueOffset);
                Add(a, a, rab, validMask);
            }
            Mul(a, a, alphaReg, validMask);
            ExpSub(z, zeros, a, validMask);
            Add(z, ones, z, validMask);
            Div(z, ones, z, validMask);
            Mul(s, z, a, validMask);
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
            Select(s, s, zeros, scoreMask);
            if constexpr (!std::is_same<Type, AccType>::value)
                CastDownStore<Type, AccType>(ubSiluScorePtr + valueOffset, s, validMask);
            else
                StoreAlign(ubSiluScorePtr + valueOffset, s, validMask);
        }
    }
}

}  // namespace Catlass::Epilogue::RegBase
