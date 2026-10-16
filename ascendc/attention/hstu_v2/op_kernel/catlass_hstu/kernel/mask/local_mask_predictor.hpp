/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

namespace Catlass::Kernel::Mask {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, typename ElementOffset>
struct LocalMaskPredictor {
    struct BlockPredParams {
        uint32_t qSeqId;
        uint32_t kSeqId;
        uint32_t seqlenQ;
        uint32_t seqlenK;
        uint32_t swizzleDir;
        uint32_t numContext;
        uint32_t numTarget;
        uint32_t targetGroupSize;
    };

    template <typename Kernel>
    CATLASS_DEVICE void Construct(Kernel* kernel, const typename Kernel::Params& params)
    {
        windowSizeLeft = kernel->windowSizeLeft;
        windowSizeRight = kernel->windowSizeRight;
        targetGroupSize = kernel->targetGroupSize;
        blockM = kernel->blockM;
        blockN = kernel->blockN;
        if constexpr (Kernel::IS_CONTEXT_V) {
            gNumContext.SetGlobalBuffer((__gm__ ElementOffset*)params.ptrNumContext);
        }
        if constexpr (Kernel::IS_TARGET_V) {
            gNumTarget.SetGlobalBuffer((__gm__ ElementOffset*)params.ptrNumTarget);
        }
    }

    template <typename Kernel, typename Coord>
    CATLASS_DEVICE BlockPredParams MakeBlockPredParams(Coord blockCoord, Kernel*, uint32_t seqlenQ, uint32_t seqlenK,
                                                       uint32_t swizzleDir)
    {
        const uint32_t batch = static_cast<uint32_t>(tla::get<0>(blockCoord));
        uint32_t context = 0U;
        uint32_t target = 0U;
        if constexpr (Kernel::IS_CONTEXT_V) {
            context = gNumContext.GetValue(batch);
        }
        if constexpr (Kernel::IS_TARGET_V) {
            target = gNumTarget.GetValue(batch);
        }
        return {static_cast<uint32_t>(tla::get<2>(blockCoord)),
                static_cast<uint32_t>(tla::get<3>(blockCoord)),
                seqlenQ,
                seqlenK,
                swizzleDir,
                context,
                target,
                targetGroupSize};
    }

    CATLASS_DEVICE void Classifier(BlockPredParams& bp)
    {
        params = bp;
        const int32_t delta = static_cast<int32_t>(bp.seqlenK) - static_cast<int32_t>(bp.seqlenQ);
        const int32_t qBegin = static_cast<int32_t>(bp.qSeqId * blockM);
        const int32_t qEnd = min(qBegin + static_cast<int32_t>(blockM), static_cast<int32_t>(bp.seqlenQ)) - 1;
        const int32_t kBegin = static_cast<int32_t>(bp.kSeqId * blockN);
        const int32_t kEnd = min(kBegin + static_cast<int32_t>(blockN), static_cast<int32_t>(bp.seqlenK)) - 1;
        const int32_t minDistance = qBegin + delta - kEnd;
        const int32_t maxDistance = qEnd + delta - kBegin;
        const int32_t lower = windowSizeRight < 0 ? INT32_MIN : -windowSizeRight;
        const int32_t upper = windowSizeLeft < 0 ? INT32_MAX : windowSizeLeft;
        outside = qBegin >= static_cast<int32_t>(bp.seqlenQ) || kBegin >= static_cast<int32_t>(bp.seqlenK) ||
                  maxDistance < lower || minDistance > upper;
        needMask = bp.numContext > 0 || bp.numTarget > 0 || outside || minDistance < lower || maxDistance > upper;
    }

    CATLASS_DEVICE bool IsSkip() const
    {
        return false;
    }
    CATLASS_DEVICE bool UsesDirectScoreMask() const
    {
        return false;
    }

    template <class Coord>
    CATLASS_DEVICE void CopyPackedMask(AscendC::LocalTensor<int32_t>&, Coord const&, uint32_t) const
    {
    }

    template <typename Elem, class Coord, class Shape>
    CATLASS_DEVICE bool ApplyMask(AscendC::LocalTensor<Elem>& mask, Coord const& coord, Shape const& shape,
                                  uint32_t alignRows, uint32_t alignCols) const
    {
        if (!needMask)
            return false;
        const int32_t qBase = static_cast<int32_t>(tla::get<2>(coord));
        const int32_t kBase = static_cast<int32_t>(tla::get<3>(coord));
        const int32_t delta = static_cast<int32_t>(params.seqlenK) - static_cast<int32_t>(params.seqlenQ);
        const int32_t rows = static_cast<int32_t>(tla::get<0>(shape));
        const int32_t cols = static_cast<int32_t>(tla::get<1>(shape));
        AscendC::NumericLimits<Elem>::NegativeInfinity(mask, alignRows * alignCols);
        for (int32_t row = 0; row < rows; ++row) {
            const int32_t center = qBase + row + delta - kBase;
            int32_t begin = windowSizeLeft < 0 ? 0 : max(0, center - windowSizeLeft);
            int32_t end = windowSizeRight < 0 ? cols : min(cols, center + windowSizeRight + 1);
            if (end > begin)
                AscendC::Duplicate<Elem>(mask[row * alignCols + begin], 0, end - begin);
        }
        return true;
    }

    CATLASS_DEVICE bool IsInnerLoopFirstQBlock(BlockPredParams& bp)
    {
        return (bp.swizzleDir == 1) ? !BlockIntersects(bp, static_cast<int32_t>(bp.qSeqId) - 1)
                                    : !BlockIntersects(bp, static_cast<int32_t>(bp.qSeqId) + 1);
    }
    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp)
    {
        return (bp.swizzleDir == 1) ? !BlockIntersects(bp, static_cast<int32_t>(bp.qSeqId) + 1)
                                    : !BlockIntersects(bp, static_cast<int32_t>(bp.qSeqId) - 1);
    }

private:
    CATLASS_DEVICE bool BlockIntersects(const BlockPredParams& bp, int32_t qBlock) const
    {
        if (qBlock < 0 || qBlock >= static_cast<int32_t>((bp.seqlenQ + blockM - 1U) / blockM))
            return false;
        if (bp.numContext > 0 || bp.numTarget > 0)
            return true;
        const int32_t delta = static_cast<int32_t>(bp.seqlenK) - static_cast<int32_t>(bp.seqlenQ);
        const int32_t qBegin = qBlock * static_cast<int32_t>(blockM);
        const int32_t qEnd = min(qBegin + static_cast<int32_t>(blockM), static_cast<int32_t>(bp.seqlenQ)) - 1;
        const int32_t kBegin = static_cast<int32_t>(bp.kSeqId * blockN);
        const int32_t kEnd = min(kBegin + static_cast<int32_t>(blockN), static_cast<int32_t>(bp.seqlenK)) - 1;
        const int32_t minDistance = qBegin + delta - kEnd;
        const int32_t maxDistance = qEnd + delta - kBegin;
        const int32_t lower = windowSizeRight < 0 ? INT32_MIN : -windowSizeRight;
        const int32_t upper = windowSizeLeft < 0 ? INT32_MAX : windowSizeLeft;
        return maxDistance >= lower && minDistance <= upper;
    }

public:
    BlockPredParams params{};
    int32_t windowSizeLeft{-1};
    int32_t windowSizeRight{-1};
    uint32_t targetGroupSize{0};
    uint32_t blockM{BLOCK_M};
    uint32_t blockN{BLOCK_N};
    bool needMask{false};
    bool outside{false};
    AscendC::GlobalTensor<ElementOffset> gNumContext;
    AscendC::GlobalTensor<ElementOffset> gNumTarget;
};

}  // namespace Catlass::Kernel::Mask
