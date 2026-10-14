/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "../../../../common/hstu_schedule_policy.h"
#include "../../../../../attn_plan/common/attn_metadata_layout.h"

namespace Catlass::Gemm::Block {

template <class ElementOffset_, uint32_t BLOCK_M, uint32_t BLOCK_N>
class StreamKOuterBlockScheduler {
public:
    using ElementOffset = ElementOffset_;

    CATLASS_DEVICE StreamKOuterBlockScheduler(uint32_t batch, uint32_t heads, GM_ADDR seqQ, GM_ADDR seqK,
                                               uint32_t blockM = BLOCK_M, uint32_t blockN = BLOCK_N,
                                               GM_ADDR metadata = nullptr)
        : batchSize(batch), headNum(heads), blockM(blockM), blockN(blockN)
    {
        seqOffsetQ.SetGlobalBuffer((__gm__ ElementOffset*)seqQ);
        seqOffsetK.SetGlobalBuffer((__gm__ ElementOffset*)seqK);
        if (metadata != nullptr) meta.SetGlobalBuffer((__gm__ int32_t*)metadata);
        hasMetadata = metadata != nullptr;
    }

    CATLASS_DEVICE void EnableTargetWorkload(GM_ADDR, GM_ADDR, uint32_t, bool) {}
    CATLASS_DEVICE void EnableCausalWorkload() {}
    CATLASS_DEVICE void EnableSingleOuterBlockFastPath(bool) {}
    CATLASS_DEVICE void EnableUniformOuterBlockFastPath(bool, uint32_t) {}
    CATLASS_DEVICE void EnablePackedSparseWorkload(GM_ADDR sparseInfo, uint32_t maskHeads, uint32_t maxOuter)
    {
        AscendC::ListTensorDesc list;
        list.Init(sparseInfo);
        auto descriptor = (__gm__ uint8_t*)list.GetDataPtr<__gm__ uint8_t>(6);
        packedState.SetGlobalBuffer((__gm__ int32_t*)descriptor);
        maskHeadNum = maskHeads;
        maxOuterBlocks = maxOuter;
        usePacked = true;
    }

    CATLASS_DEVICE void Init()
    {
        uint32_t coreId = 0;
        if ASCEND_IS_AIV {
            coreId = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            coreId = AscendC::GetBlockIdx();
        }
        const uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t totalTasks = 0;
        for (uint32_t b = 0; b < batchSize; ++b) {
            const uint32_t qLen = seqOffsetQ.GetValue(b + 1U) - seqOffsetQ.GetValue(b);
            const uint32_t kLen = seqOffsetK.GetValue(b + 1U) - seqOffsetK.GetValue(b);
            totalTasks += headNum * RuntimeCeilDiv(qLen, blockM) * RuntimeCeilDiv(kLen, blockN);
        }
        if (hasMetadata && static_cast<uint32_t>(meta.GetValue(ops_rec::attn_plan::metadata::HEAD_DIRECT_MAGIC)) ==
                               ops_rec::attn_plan::metadata::STREAM_K_MAGIC) {
            const uint32_t sectionNum = static_cast<uint32_t>(
                meta.GetValue(ops_rec::attn_plan::metadata::HEAD_SECTION_NUM));
            const uint32_t encodedCoreNum = static_cast<uint32_t>(
                meta.GetValue(ops_rec::attn_plan::metadata::HEAD_AIC_CORE_NUM));
            uint32_t count = 0U;
            currentTask = 0U;
            bool found = false;
            for (uint32_t section = 0; section < sectionNum; ++section) {
                const uint32_t record = ops_rec::attn_plan::metadata::HEAD_STRIDE +
                    (section * encodedCoreNum + coreId) * ops_rec::attn_plan::metadata::FA_STRIDE;
                const uint32_t sectionCount = static_cast<uint32_t>(
                    meta.GetValue(record + ops_rec::attn_plan::metadata::DIRECT_TASK_COUNT));
                if (!found && sectionCount != 0U) {
                    currentTask = static_cast<uint32_t>(
                        meta.GetValue(record + ops_rec::attn_plan::metadata::DIRECT_TASK_BEGIN));
                    found = true;
                }
                count += sectionCount;
            }
            taskEnd = currentTask + count;
        } else {
            const auto range = hstu::schedule::UniformCoreRange(totalTasks, coreId, coreNum);
            currentTask = range.beginBlock;
            taskEnd = range.endBlock;
        }
        Locate(currentTask);
        AdvanceToExecutable();
        Update();
    }

    CATLASS_DEVICE bool IsValid() const { return currentTask < taskEnd; }
    CATLASS_DEVICE bool IsLast() const { return currentTask + 1U == taskEnd; }
    CATLASS_DEVICE bool IsFirstInOuter() const { return firstInOuter; }
    CATLASS_DEVICE bool IsLastInOuter() const
    {
        if (!IsValid()) return false;
        const uint32_t rowEnd = currentTask + (kBlockCount - kBlockId);
        const uint32_t scanEnd = min(taskEnd, rowEnd);
        for (uint32_t next = currentTask + 1U; next < scanEnd; ++next) {
            if (IsExecutableTask(next)) return false;
        }
        return true;
    }
    CATLASS_DEVICE uint32_t GetHeadBlockId() const { return qBlockId; }
    CATLASS_DEVICE uint32_t GetCurrentSeqLen() const { return qSeqLen; }
    CATLASS_DEVICE uint32_t GetKBlockId() const { return kBlockId; }
    CATLASS_DEVICE uint32_t GetKSeqLen() const { return kSeqLen; }
    CATLASS_DEVICE uint32_t GetKBaseOffset() const { return kBaseOffset; }

    CATLASS_DEVICE auto GetMeta() const { return tla::MakeCoord(batchId, headId, qBlockId); }

    template <class Tensor>
    CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        return tla::GetTile(tensor, tla::MakeCoord(qOffset, 0), tla::MakeShape(qBlockSize, dim));
    }

    template <class Tensor>
    CATLASS_DEVICE auto GetShareTile(Tensor const& tensor, int64_t totalSeq)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(headId * totalSeq + qBaseOffset + qBlockId * blockM, 0);
        return tla::GetTile(tensor, coord, tla::MakeShape(qBlockSize, dim));
    }

    CATLASS_DEVICE StreamKOuterBlockScheduler& operator++()
    {
        const uint32_t previousBatch = batchId;
        const uint32_t previousHead = headId;
        const uint32_t previousOuter = qBlockId;
        ++currentTask;
        Locate(currentTask);
        AdvanceToExecutable();
        firstInOuter = !IsValid() || batchId != previousBatch || headId != previousHead || qBlockId != previousOuter;
        Update();
        return *this;
    }

private:
    CATLASS_DEVICE void Locate(uint32_t flat)
    {
        for (uint32_t b = 0; b < batchSize; ++b) {
            const uint32_t qLen = seqOffsetQ.GetValue(b + 1U) - seqOffsetQ.GetValue(b);
            const uint32_t kLen = seqOffsetK.GetValue(b + 1U) - seqOffsetK.GetValue(b);
            const uint32_t qBlocks = RuntimeCeilDiv(qLen, blockM);
            const uint32_t kBlocks = RuntimeCeilDiv(kLen, blockN);
            const uint32_t batchTasks = headNum * qBlocks * kBlocks;
            if (flat < batchTasks) {
                batchId = b; qSeqLen = qLen; kSeqLen = kLen;
                qBaseOffset = seqOffsetQ.GetValue(b); kBaseOffset = seqOffsetK.GetValue(b);
                const uint32_t perHead = qBlocks * kBlocks;
                headId = flat / perHead;
                const uint32_t inHead = flat - headId * perHead;
                qBlockId = inHead / kBlocks;
                kBlockId = inHead - qBlockId * kBlocks;
                kBlockCount = kBlocks;
                return;
            }
            flat -= batchTasks;
        }
        batchId = batchSize;
    }

    CATLASS_DEVICE bool IsExecutable() const
    {
        if (!usePacked || currentTask >= taskEnd || batchId >= batchSize) return true;
        const uint32_t maskHead = maskHeadNum == 1U ? 0U : headId;
        const uint32_t qBlocks = RuntimeCeilDiv(qSeqLen, blockM);
        const uint32_t kBlocks = RuntimeCeilDiv(kSeqLen, blockN);
        const uint32_t outer = (batchId * maskHeadNum + maskHead) * maxOuterBlocks + qBlockId;
        const uint32_t state = static_cast<uint32_t>(packedState.GetValue(outer * kBlocks + kBlockId)) & 0x3U;
        return state == 0x1U || state == 0x3U;
    }

    CATLASS_DEVICE bool IsExecutableTask(uint32_t flat) const
    {
        if (!usePacked) return true;
        uint32_t value = flat;
        for (uint32_t b = 0; b < batchSize; ++b) {
            const uint32_t qLen = seqOffsetQ.GetValue(b + 1U) - seqOffsetQ.GetValue(b);
            const uint32_t kLen = seqOffsetK.GetValue(b + 1U) - seqOffsetK.GetValue(b);
            const uint32_t qBlocks = RuntimeCeilDiv(qLen, blockM);
            const uint32_t kBlocks = RuntimeCeilDiv(kLen, blockN);
            const uint32_t perHead = qBlocks * kBlocks;
            const uint32_t batchTasks = headNum * perHead;
            if (value < batchTasks) {
                const uint32_t h = value / perHead;
                const uint32_t inHead = value - h * perHead;
                const uint32_t outerBlock = inHead / kBlocks;
                const uint32_t innerBlock = inHead - outerBlock * kBlocks;
                const uint32_t maskHead = maskHeadNum == 1U ? 0U : h;
                const uint32_t outer = (b * maskHeadNum + maskHead) * maxOuterBlocks + outerBlock;
                const uint32_t state = static_cast<uint32_t>(
                    packedState.GetValue(outer * kBlocks + innerBlock)) & 0x3U;
                return state == 0x1U || state == 0x3U;
            }
            value -= batchTasks;
        }
        return false;
    }

    CATLASS_DEVICE void AdvanceToExecutable()
    {
        while (currentTask < taskEnd && !IsExecutable()) {
            ++currentTask;
            Locate(currentTask);
        }
    }

    CATLASS_DEVICE void Update()
    {
        if (!IsValid() || batchId >= batchSize) { qBlockSize = 0; qOffset = 0; return; }
        const uint32_t local = qBlockId * blockM;
        qBlockSize = min(blockM, qSeqLen - local);
        qOffset = (qBaseOffset + local) * headNum + headId;
    }

    uint32_t batchSize{0}, headNum{0}, blockM{BLOCK_M}, blockN{BLOCK_N};
    uint32_t currentTask{0}, taskEnd{0};
    uint32_t batchId{0}, headId{0}, qBlockId{0}, kBlockId{0};
    uint32_t qSeqLen{0}, kSeqLen{0}, qBaseOffset{0}, kBaseOffset{0}, kBlockCount{0};
    uint32_t qBlockSize{0}, qOffset{0};
    AscendC::GlobalTensor<ElementOffset> seqOffsetQ, seqOffsetK;
    AscendC::GlobalTensor<int32_t> meta;
    AscendC::GlobalTensor<int32_t> packedState;
    uint32_t maskHeadNum{0}, maxOuterBlocks{0};
    bool usePacked{false};
    bool hasMetadata{false};
    bool firstInOuter{true};
};

template <class Outer_, uint32_t BLOCK_M, uint32_t BLOCK_N, bool SPLIT_RUNTIME_INNER = false>
class StreamKInnerBlockScheduler {
public:
    using Outer = Outer_;
    using ElementOffset = typename Outer::ElementOffset;
    CATLASS_DEVICE StreamKInnerBlockScheduler(uint32_t, uint32_t heads, GM_ADDR seqK,
                                               uint32_t blockM = BLOCK_M, uint32_t blockN = BLOCK_N)
        : headNum(heads), blockM(blockM), blockN(blockN)
    {
        seqOffsetK.SetGlobalBuffer((__gm__ ElementOffset*)seqK);
    }
    CATLASS_DEVICE void EnablePackedSparse(GM_ADDR sparseInfo, uint32_t maskHeads, uint32_t maxOuter,
                                           uint32_t maxInner)
    {
        AscendC::ListTensorDesc list;
        list.Init(sparseInfo);
        auto descriptor = (__gm__ uint8_t*)list.GetDataPtr<__gm__ uint8_t>(6);
        packedState.SetGlobalBuffer((__gm__ int32_t*)descriptor);
        maskHeadNum = maskHeads; maxOuterBlocks = maxOuter; maxInnerBlocks = maxInner; usePacked = true;
    }
    CATLASS_DEVICE void EnableCausal(GM_ADDR, GM_ADDR, uint32_t, bool, bool) {}
    CATLASS_DEVICE void EnableLocal(int32_t, int32_t, bool) {}
    CATLASS_DEVICE void Init(Outer const& outer)
    {
        auto meta = outer.GetMeta();
        batchId = tla::get<0>(meta); headId = tla::get<1>(meta); rowBlockId = tla::get<2>(meta);
        blockId = outer.GetKBlockId(); seqLens = outer.GetKSeqLen(); blockBaseOffset = outer.GetKBaseOffset();
        valid = outer.IsValid();
        outerFirst = outer.IsFirstInOuter();
        outerLast = outer.IsLastInOuter();
        subBlockId = 0U;
        if (valid && usePacked) {
            const uint32_t maskHead = maskHeadNum == 1U ? 0U : headId;
            const uint32_t outerPos = (batchId * maskHeadNum + maskHead) * maxOuterBlocks + rowBlockId;
            const uint32_t state = static_cast<uint32_t>(
                packedState.GetValue(outerPos * maxInnerBlocks + static_cast<uint32_t>(blockId))) & 0x3U;
            valid = state == 0x1U || state == 0x3U;
        }
        Update();
    }
    CATLASS_DEVICE bool GetTriggerSwizzle() const { return false; }
    CATLASS_DEVICE uint32_t GetSwizzlePosition() const { return 0; }
    CATLASS_DEVICE bool IsFirst() const { return valid && outerFirst && subBlockId == 0U; }
    CATLASS_DEVICE bool IsLast() const { return valid && outerLast && subBlockId + 1U == subBlockCount; }
    CATLASS_DEVICE bool IsValid() const { return valid; }
    CATLASS_DEVICE int32_t GetBlockId() const { return blockId; }
    CATLASS_DEVICE uint32_t GetCacheBlockId() const
    {
        const uint32_t chunksPerLogical = SPLIT_RUNTIME_INNER ? RuntimeCeilDiv(blockN, BLOCK_N) : 1U;
        return static_cast<uint32_t>(blockId) * chunksPerLogical + subBlockId;
    }
    CATLASS_DEVICE bool IsFirstSubBlock() const { return subBlockId == 0U; }
    CATLASS_DEVICE bool IsLastSubBlock() const { return subBlockId + 1U == subBlockCount; }
    CATLASS_DEVICE uint32_t GetRowBlockId() const { return rowBlockId; }
    CATLASS_DEVICE uint32_t GetSeqLens() const { return seqLens; }
    CATLASS_DEVICE uint32_t GetSwizzleDir() const { return 1; }
    CATLASS_DEVICE StreamKInnerBlockScheduler& operator++()
    {
        if (subBlockId + 1U < subBlockCount) { ++subBlockId; Update(); } else { valid = false; }
        return *this;
    }
    CATLASS_DEVICE void SkipBlocks(uint32_t) { valid = false; }
    template <class Tensor> CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        return tla::GetTile(tensor, tla::MakeCoord(blockOffset + headId, 0), tla::MakeShape(blockSize, dim));
    }
    template <class Tensor> CATLASS_DEVICE auto GetShareTile(Tensor const& tensor, int64_t totalSeq)
    {
        auto dim = tla::get<1>(tensor.shape());
        return tla::GetTile(tensor, tla::MakeCoord(headId * totalSeq + blockOffset / headNum, 0),
                            tla::MakeShape(blockSize, dim));
    }
    template <class Coord, class Shape> CATLASS_DEVICE auto GetTileMapping(Coord const&, Shape const& qShape)
    {
        const uint32_t physicalN = SPLIT_RUNTIME_INNER ? BLOCK_N : blockN;
        return tla::MakeTuple(tla::MakeCoord(batchId, headId, blockId * blockN + subBlockId * physicalN,
                                             rowBlockId * blockM),
                              tla::MakeShape(blockSize, tla::get<0>(qShape)));
    }
private:
    CATLASS_DEVICE void Update()
    {
        if (!valid) { blockSize = 0; blockOffset = 0; return; }
        const uint32_t logical = static_cast<uint32_t>(blockId) * blockN;
        const uint32_t logicalSize = min(blockN, seqLens - logical);
        const uint32_t physicalN = SPLIT_RUNTIME_INNER ? BLOCK_N : blockN;
        subBlockCount = RuntimeCeilDiv(logicalSize, physicalN);
        const uint32_t local = logical + subBlockId * physicalN;
        blockSize = min(physicalN, logicalSize - subBlockId * physicalN);
        blockOffset = (blockBaseOffset + local) * headNum;
    }
    uint32_t headNum{0}, blockM{BLOCK_M}, blockN{BLOCK_N};
    uint32_t batchId{0}, headId{0}, rowBlockId{0}, seqLens{0}, blockBaseOffset{0};
    int32_t blockId{0}; uint32_t blockSize{0}, blockOffset{0}; bool valid{false};
    uint32_t subBlockId{0}, subBlockCount{1};
    bool outerFirst{false}, outerLast{false};
    AscendC::GlobalTensor<ElementOffset> seqOffsetK;
    AscendC::GlobalTensor<int32_t> packedState;
    uint32_t maskHeadNum{0}, maxOuterBlocks{0}, maxInnerBlocks{0};
    bool usePacked{false};
};

}  // namespace Catlass::Gemm::Block
