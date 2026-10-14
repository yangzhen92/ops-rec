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
 * @file arbitrary_mask_predictor.hpp
 * @brief Packed-only arbitrary-mask block classifier and bitmap loader.
 *
 * HSTU consumes the direct descriptor format produced by packed_block_sparse:
 *   descriptor = (inner_block << 2) | state
 *   state: 0 = empty, 1 = full, 3 = partial.
 * PARTIAL blocks use a fixed row-local bitmap slot, so classification and
 * bitmap lookup are both O(1). Legacy six-tensor index lists, compressed state
 * words, partial-prefix scans and arbitrary_func mask reconstruction are not
 * part of this hot path.
 */

#pragma once

namespace Catlass::Kernel::Mask {

constexpr size_t SPARSE_INFO_TENSOR_NUM = 9;

enum SparseInfoIndex : size_t {
    MASK_CNT = 0,
    ROW_BOUNDS = 1,
    FULL_CNT = 3,
    DIRECT_DESCRIPTOR = 6,
    DIRECT_BITMAP = 8,
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, bool IS_FWD = true>
struct ArbitraryMaskPredictor {

    struct SparseBounds {
        int32_t firstBlk{-1};
        int32_t lastBlk{-1};
    };

    struct BlockPredParams {
        int32_t batchId{-1};
        int32_t headId{-1};
        int32_t qSeqId{-1};
        int32_t kSeqId{-1};
        int32_t qSeqLen{-1};
        int32_t kSeqLen{-1};
        uint32_t swizzleDir{0};
        SparseBounds sparse{};
        bool isMaskBlock{false};
        bool isFullBlock{false};
        int32_t bitmapSlot{-1};
    };

    template <typename Kernel>
    CATLASS_DEVICE void Construct(Kernel* kernel, const typename Kernel::Params& params)
    {
        maskHeads = kernel->maskHeads;
        blockM = kernel->blockM;
        blockN = kernel->blockN;
        AscendC::ListTensorDesc sparseListDesc;
        sparseListDesc.Init(params.ptrSparseInfo);
        auto rowBoundsPtr = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(ROW_BOUNDS);
        auto descriptorPtr = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(DIRECT_DESCRIPTOR);
        auto bitmapPtr = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(DIRECT_BITMAP);
        gRowBounds.SetGlobalBuffer((__gm__ int32_t*)rowBoundsPtr);
        gDescriptor.SetGlobalBuffer((__gm__ int32_t*)descriptorPtr);
        gBitmap.SetGlobalBuffer((__gm__ int32_t*)bitmapPtr);

        maxBlkCntQ = (kernel->maxSeqLenQ + blockM - 1U) / blockM;
        maxBlkCntK = (kernel->maxSeqLenK + blockN - 1U) / blockN;
    }

    template <typename Kernel, typename Coord>
    CATLASS_DEVICE BlockPredParams MakeBlockPredParams(Coord blockCoord, Kernel*, uint32_t seqlenQ,
                                                       uint32_t seqlenK, uint32_t swizzleDir)
    {
        BlockPredParams bp;
        bp.batchId = tla::get<0>(blockCoord);
        bp.headId = tla::get<1>(blockCoord);
        bp.qSeqId = tla::get<2>(blockCoord);
        bp.kSeqId = tla::get<3>(blockCoord);
        bp.qSeqLen = seqlenQ;
        bp.kSeqLen = seqlenK;
        bp.swizzleDir = swizzleDir;

        const uint32_t outerPos = GetOuterPos(bp);
        if (IsSparseCached(bp)) {
            bp.sparse = params.sparse;
        } else {
            bp.sparse.firstBlk = gRowBounds.GetValue(outerPos * 2);
            bp.sparse.lastBlk = gRowBounds.GetValue(outerPos * 2 + 1);
        }

        const uint32_t innerBlock = IS_FWD ? bp.kSeqId : bp.qSeqId;
        const uint32_t innerBlocks = IS_FWD ? maxBlkCntK : maxBlkCntQ;
        const uint32_t descriptor =
            static_cast<uint32_t>(gDescriptor.GetValue(outerPos * innerBlocks + innerBlock));
        const uint32_t state = descriptor & 0x3U;
        bp.isMaskBlock = state == 0x3U;
        bp.isFullBlock = state == 0x1U;
        if (bp.isMaskBlock) {
            // Direct descriptors use a fixed slot per (outer, inner) pair.
            bp.bitmapSlot = static_cast<int32_t>(outerPos * innerBlocks + (descriptor >> 2));
        }
        return bp;
    }

    CATLASS_DEVICE void Classifier(BlockPredParams& bp)
    {
        params = bp;
        isMaskBlock = bp.isMaskBlock;
        isFullBlock = bp.isFullBlock;
        isEmptyBlock = !bp.isMaskBlock && !bp.isFullBlock;
        needMask = bp.isMaskBlock;
    }

    CATLASS_DEVICE bool IsSkip() const
    {
        return isEmptyBlock;
    }

    template <class Coord>
    CATLASS_DEVICE void CopyPackedMask(AscendC::LocalTensor<int32_t>& dst, Coord const& coord,
                                       uint32_t alignRows) const
    {
        const uint32_t q = IS_FWD ? tla::get<3>(coord) : tla::get<2>(coord);
        const uint32_t rowInBlock = q - static_cast<uint32_t>(params.qSeqId) * blockM;
        const uint32_t bitmapWordsPerRow = blockN / 32U;
        const uint32_t bitmapBase =
            (static_cast<uint32_t>(params.bitmapSlot) * blockM + rowInBlock) * bitmapWordsPerRow;
        const uint32_t slotsPerRow = (blockN + 63U) / 64U;
        const uint32_t blockCount = alignRows * slotsPerRow;
        // One 64-bit compact fragment is expanded by plds(DIST_US). Put each
        // fragment in its own 32-byte-aligned UB slot with one strided DMA.
        if (blockN == 32U) {
            const AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount), 4, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 7, 0};
            AscendC::DataCopyPad(dst, gBitmap[bitmapBase], copyParams, padParams);
        } else if (blockN == 128U) {
            const AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 6, 0};
            const uint16_t first = blockCount > 128 ? 128 : static_cast<uint16_t>(blockCount);
            const AscendC::DataCopyExtParams firstCopy{first, 8, 0, 0, 0};
            AscendC::DataCopyPad(dst, gBitmap[bitmapBase], firstCopy, padParams);
            if (blockCount > first) {
                const AscendC::DataCopyExtParams secondCopy{static_cast<uint16_t>(blockCount - first), 8, 0, 0, 0};
                AscendC::DataCopyPad(dst[first * 8], gBitmap[bitmapBase + first * 2], secondCopy, padParams);
            }
        } else {
            const AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 6, 0};
            const AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount), 8, 0, 0, 0};
            AscendC::DataCopyPad(dst, gBitmap[bitmapBase], copyParams, padParams);
        }
    }

    CATLASS_DEVICE uint32_t GetPackedMaskStrideWords() const
    {
        return ((blockN + 63U) / 64U) * 8U;
    }

    CATLASS_DEVICE bool IsInnerLoopFirstQBlock(BlockPredParams& bp) const
    {
        if constexpr (IS_FWD) {
            return bp.swizzleDir == 1 ? bp.qSeqId == 0 : bp.qSeqId == (bp.qSeqLen + blockM - 1U) / blockM - 1U;
        }
        return bp.swizzleDir == 1 ? bp.qSeqId == bp.sparse.firstBlk : bp.qSeqId == bp.sparse.lastBlk;
    }

    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp) const
    {
        if constexpr (IS_FWD) {
            return bp.swizzleDir == 1 ? bp.qSeqId == (bp.qSeqLen + blockM - 1U) / blockM - 1U : bp.qSeqId == 0;
        }
        return bp.swizzleDir == 1 ? bp.qSeqId == bp.sparse.lastBlk : bp.qSeqId == bp.sparse.firstBlk;
    }

    BlockPredParams params{};
    bool needMask{false};
    bool isMaskBlock{false};
    bool isFullBlock{false};
    bool isEmptyBlock{false};

private:
    CATLASS_DEVICE bool IsSparseCached(const BlockPredParams& bp) const
    {
        if constexpr (IS_FWD) {
            return bp.batchId == params.batchId && bp.headId == params.headId && bp.qSeqId == params.qSeqId;
        }
        return bp.batchId == params.batchId && bp.headId == params.headId && bp.kSeqId == params.kSeqId;
    }

    CATLASS_DEVICE uint32_t GetOuterPos(const BlockPredParams& bp) const
    {
        const uint32_t maskHead = maskHeads == 1 ? 0 : static_cast<uint32_t>(bp.headId);
        const uint32_t batchHead = static_cast<uint32_t>(bp.batchId) * maskHeads + maskHead;
        return IS_FWD ? batchHead * maxBlkCntQ + static_cast<uint32_t>(bp.qSeqId)
                      : batchHead * maxBlkCntK + static_cast<uint32_t>(bp.kSeqId);
    }

    uint32_t maxBlkCntQ{0};
    uint32_t maxBlkCntK{0};
    uint32_t blockM{BLOCK_M};
    uint32_t blockN{BLOCK_N};
    uint32_t maskHeads{0};
    AscendC::GlobalTensor<int32_t> gRowBounds;
    AscendC::GlobalTensor<int32_t> gDescriptor;
    AscendC::GlobalTensor<int32_t> gBitmap;
};

}  // namespace Catlass::Kernel::Mask
