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
 • @file block_scheduler.hpp

 • @brief HSTU GEMM 块调度器实现

 • @description 提供行方向和列方向的块调度器，用于将计算任务分配到多个 AI Core 上

 •              支持静态负载均衡和持久化分块策略

 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "../../detail/target_block_range.hpp"
#include "../../../../common/hstu_schedule_policy.h"

namespace Catlass::Gemm::Block {
CATLASS_DEVICE uint32_t RuntimeCeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

/**
 • @brief 行方向块调度器

 • @tparam ElementOffset_ 序列偏移量的数据类型

 • @tparam BLOCK_M 行方向的块大小

 • @tparam BLOCK_N 列方向的块大小

 • @description 负责在行方向（head 维度）上进行任务分片；所有 mask

 •              模式都通过公共 workload range builder 生成连续区间。

 */
template <class ElementOffset_, uint32_t BLOCK_M, uint32_t BLOCK_N>
class OuterBlockScheduler {
public:
    using ElementOffset = ElementOffset_;

    /**
     ◦ @brief 构造函数

     ◦ @param batchSize 批次大小

     ◦ @param headNum 注意力头数

     ◦ @param seqOffsetM M 方向的序列偏移量指针

     ◦ @param seqOffsetN N 方向的序列偏移量指针

     ◦ @description 初始化调度器，设置批次大小、头数，并加载序列偏移量

     */
    CATLASS_DEVICE
    OuterBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffsetM, GM_ADDR seqOffsetN,
                      uint32_t runtimeBlockM = BLOCK_M, uint32_t runtimeBlockN = BLOCK_N)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->blockM = runtimeBlockM;
        this->blockN = runtimeBlockN;

        this->seqOffsetM.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetM);
        this->seqOffsetN.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetN);
    }

    CATLASS_DEVICE
    void EnableTargetWorkload(GM_ADDR numContext, GM_ADDR numTarget, uint32_t targetGroupSize, bool hasContext)
    {
        useTargetWorkload = true;
        this->targetGroupSize = targetGroupSize;
        this->hasContext = hasContext;
        this->numTarget.SetGlobalBuffer((__gm__ ElementOffset*)numTarget);
        if (hasContext) {
            this->numContext.SetGlobalBuffer((__gm__ ElementOffset*)numContext);
        }
    }

    CATLASS_DEVICE void EnableCausalWorkload()
    {
        useCausalWorkload = true;
    }

    CATLASS_DEVICE void EnableSingleOuterBlockFastPath(bool enable)
    {
        useSingleOuterBlockFastPath = enable;
    }

    CATLASS_DEVICE void EnableUniformOuterBlockFastPath(bool enable, uint32_t maxSeqLen)
    {
        useUniformOuterBlockFastPath = enable;
        uniformOuterSeqLen = maxSeqLen;
    }

    CATLASS_DEVICE
    void EnablePackedSparseWorkload(GM_ADDR sparseInfo, uint32_t maskHeadNum, uint32_t maxOuterBlocks)
    {
        if (sparseInfo == nullptr || maskHeadNum == 0) {
            return;
        }
        AscendC::ListTensorDesc sparseListDesc;
        sparseListDesc.Init(sparseInfo);
        GM_ADDR maskCntPtr =
            (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(0);
        GM_ADDR fullCntPtr =
            (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(3);
        this->maskBlockCnt.SetGlobalBuffer((__gm__ int32_t*)maskCntPtr);
        this->fullBlockCnt.SetGlobalBuffer((__gm__ int32_t*)fullCntPtr);
        this->maskHeadNum = maskHeadNum;
        this->sparseOuterStride = maxOuterBlocks;
        this->usePackedSparseWorkload = true;
    }

    /**
     ◦ @brief 初始化块信息

     ◦ @param coreStartBlockId 核心起始块 ID

     ◦ @description 根据起始块 ID 计算对应的 batchId、headId、headBlockId 等信息

     */
    CATLASS_DEVICE
    void InitBlock(uint32_t coreStartBlockId)
    {
        uint32_t blockOffset = 0;
        for (auto b = 0; b < this->batchSize; ++b) {
            auto seqLens = (seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b));
            auto mBlockCnt = RuntimeCeilDiv(seqLens, blockM);
            blockOffset += (mBlockCnt * this->headNum);
            if (coreStartBlockId < blockOffset) {
                this->batchId = b;
                coreStartBlockId -= (blockOffset - mBlockCnt * this->headNum);
                this->headId = coreStartBlockId / mBlockCnt;
                this->headBlockId = coreStartBlockId - mBlockCnt * this->headId;
                this->currentSeqLen = seqLens;
                this->headBlockCnt = mBlockCnt;
                this->batchBaseOffset = seqOffsetM.GetValue(b);
                break;
            }
        }
    }

    CATLASS_DEVICE uint32_t GetSparseRowWorkload(uint32_t batch, uint32_t head, uint32_t rowBlock) const
    {
        const uint32_t maskHead = maskHeadNum == 1 ? 0 : head;
        const uint32_t sparseRow = (batch * maskHeadNum + maskHead) * sparseOuterStride + rowBlock;
        return static_cast<uint32_t>(maskBlockCnt.GetValue(sparseRow)) +
               static_cast<uint32_t>(fullBlockCnt.GetValue(sparseRow));
    }

    CATLASS_DEVICE
    uint32_t GetTargetQBlockEnd(uint32_t kBlockId, uint32_t seqLenM, uint32_t seqLenN, uint32_t targetNum,
                                uint32_t contextNum)
    {
        // This scheduler's outer row is K, so its BLOCK_M is the K-axis tile
        // and BLOCK_N is the Q-axis tile. The helper names parameters by
        // semantic Q/K axes, hence the intentional template/length swap.
        return Catlass::Detail::GetTargetQBlockEndRuntime(kBlockId, seqLenN, seqLenM, targetNum,
            hasContext ? contextNum : 0, targetGroupSize, blockN, blockM);
    }

    CATLASS_DEVICE
    uint32_t GetKBlockWorkload(uint32_t kBlockId, uint32_t seqLenM, uint32_t seqLenN, uint32_t targetNum,
                               uint32_t contextNum)
    {
        const uint32_t qBlockEnd = GetTargetQBlockEnd(kBlockId, seqLenM, seqLenN, targetNum, contextNum);
        const int64_t qkOffset = static_cast<int64_t>(seqLenM) - seqLenN;
        const int64_t firstCausalQElement =
            static_cast<int64_t>(kBlockId) * blockM - qkOffset - (static_cast<int64_t>(blockN) - 1);
        uint32_t qBlockBegin = 0;
        if (firstCausalQElement > 0) {
            qBlockBegin = static_cast<uint32_t>((firstCausalQElement + blockN - 1) / blockN);
        }

        uint32_t workload = qBlockEnd > qBlockBegin ? qBlockEnd - qBlockBegin : 0;
        if (hasContext) {
            const uint32_t historyKBlocks = RuntimeCeilDiv(seqLenM - targetNum, blockM);
            if (kBlockId < historyKBlocks) {
                const uint32_t contextQBlocks = RuntimeCeilDiv(contextNum, blockN);
                workload += contextQBlocks < qBlockBegin ? contextQBlocks : qBlockBegin;
            }
        }
        return workload;
    }

    /**
     ◦ @brief 初始化调度器

     ◦ @description 获取当前核心 ID 和总核心数，根据批次大小选择合适的分片策略

     */
    CATLASS_DEVICE
    void Init()
    {
        uint32_t coreId = 0;
        if ASCEND_IS_AIV {
            coreId = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            coreId = AscendC::GetBlockIdx();
        }
        uint32_t coreNum = AscendC::GetBlockNum();

        UnifiedSplitCore(coreId, coreNum);

        Update();
    }

    /**
     ◦ @brief 判断是否是最后一个块 (OuterBlockScheduler)

     ◦ @return bool 如果剩余块数为 1 则返回 true

     */
    CATLASS_DEVICE
    bool IsLast()
    {
        return this->blockCnt == 1;
    }

    /**
     ◦ @brief 判断块调度是否有效

     ◦ @return bool 如果块数不为 0 则返回 true

     */
    CATLASS_DEVICE
    bool IsValid()
    {
        return !(this->blockCnt == 0);
    }

    CATLASS_DEVICE uint32_t GetHeadBlockId() const
    {
        return headBlockId;
    }
    CATLASS_DEVICE uint32_t GetCurrentSeqLen() const
    {
        return currentSeqLen;
    }

    /**
     ◦ @brief 前缀递增运算符，移动到下一个块

     ◦ @return OuterBlockScheduler& 调度器引用

     ◦ @description 更新 batchId、headId、headBlockId 等信息，移动到下一个计算块

     */
    CATLASS_DEVICE
    OuterBlockScheduler& operator++()
    {
        this->blockCnt--;
        headBlockId++;

        if (headBlockId >= headBlockCnt) {
            headBlockId = 0;
            headId++;

            if (headId >= headNum) {
                headId = 0;
                batchBaseOffset += currentSeqLen;
                batchId++;
                if (batchId < batchSize) {
                    currentSeqLen = seqOffsetM.GetValue(batchId + 1) - seqOffsetM.GetValue(batchId);
                    headBlockCnt = RuntimeCeilDiv(currentSeqLen, blockM);
                }
            }
        }

        Update();
        return *this;
    }

    CATLASS_DEVICE
    auto GetMeta() const
    {
        return tla::MakeCoord(batchId, headId, headBlockId);
    }

    template <class Tensor>
    CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(this->blockOffset, 0);
        auto shape = tla::MakeShape(this->blockSize, dim);
        return tla::GetTile(tensor, coord, shape);
    }

private:
    CATLASS_DEVICE uint32_t GetTotalRows(uint32_t& rowsPerBatch, bool& uniformRows) const
    {
        uint32_t totalRows = 0;
        rowsPerBatch = 0U;
        uniformRows = true;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            const uint32_t seqLen = seqOffsetM.GetValue(batch + 1) - seqOffsetM.GetValue(batch);
            const uint32_t rows = RuntimeCeilDiv(seqLen, blockM);
            if (batch == 0U) rowsPerBatch = rows;
            else if (rows != rowsPerBatch) uniformRows = false;
            totalRows += rows * headNum;
        }
        return totalRows;
    }

    CATLASS_DEVICE void InitUniformBlock(uint32_t flatBlock, uint32_t rowsPerBatch)
    {
        const uint32_t blocksPerBatch = rowsPerBatch * headNum;
        batchId = flatBlock / blocksPerBatch;
        const uint32_t localBlock = flatBlock - batchId * blocksPerBatch;
        headId = localBlock / rowsPerBatch;
        headBlockId = localBlock - headId * rowsPerBatch;
        headBlockCnt = rowsPerBatch;
        batchBaseOffset = seqOffsetM.GetValue(batchId);
        currentSeqLen = seqOffsetM.GetValue(batchId + 1U) - batchBaseOffset;
    }

    CATLASS_DEVICE uint64_t GetRowWorkload(uint32_t batch, uint32_t head, uint32_t row,
                                            bool denseFallback)
    {
        const uint32_t seqLenM = seqOffsetM.GetValue(batch + 1) - seqOffsetM.GetValue(batch);
        const uint32_t seqLenN = seqOffsetN.GetValue(batch + 1) - seqOffsetN.GetValue(batch);
        if (useTargetWorkload || useCausalWorkload) {
            const uint32_t targetNum = useTargetWorkload ? numTarget.GetValue(batch) : 0U;
            const uint32_t contextNum = useTargetWorkload && hasContext ? numContext.GetValue(batch) : 0U;
            return GetKBlockWorkload(row, seqLenM, seqLenN, targetNum, contextNum);
        }
        if (usePackedSparseWorkload && !denseFallback) {
            return GetSparseRowWorkload(batch, head, row);
        }
        return RuntimeCeilDiv(seqLenN, blockN);
    }

    CATLASS_DEVICE void UnifiedSplitCore(uint32_t coreId, uint32_t coreNum)
    {
        if (useSingleOuterBlockFastPath) {
            const uint32_t totalBlocks = batchSize * headNum;
            const auto range = hstu::schedule::UniformCoreRange(totalBlocks, coreId, coreNum);
            blockCnt = range.endBlock - range.beginBlock;
            if (blockCnt != 0U) {
                const uint32_t flatBlock = range.beginBlock;
                batchId = flatBlock / headNum;
                headId = flatBlock - batchId * headNum;
                headBlockId = 0U;
                headBlockCnt = 1U;
                batchBaseOffset = seqOffsetM.GetValue(batchId);
                currentSeqLen = seqOffsetM.GetValue(batchId + 1U) - batchBaseOffset;
            } else {
                batchId = batchSize;
            }
            return;
        }
        if (useUniformOuterBlockFastPath && !usePackedSparseWorkload && batchSize != 0U) {
            const auto begin = seqOffsetM.GetValue(0U);
            const auto end = seqOffsetM.GetValue(batchSize);
            const uint64_t expectedEnd = static_cast<uint64_t>(batchSize) * uniformOuterSeqLen;
            if (begin == 0 && end >= 0 && static_cast<uint64_t>(end) == expectedEnd) {
                const uint32_t rowsPerBatch = RuntimeCeilDiv(uniformOuterSeqLen, blockM);
                const uint32_t totalRows = rowsPerBatch * headNum * batchSize;
                if (totalRows >= coreNum * 4U) {
                    const auto range = hstu::schedule::UniformCoreRange(totalRows, coreId, coreNum);
                    blockCnt = range.endBlock - range.beginBlock;
                    if (blockCnt != 0U) InitUniformBlock(range.beginBlock, rowsPerBatch);
                    else batchId = batchSize;
                    return;
                }
            }
        }
        uint32_t rowsPerBatch = 0U;
        bool uniformRows = false;
        const uint32_t totalRows = GetTotalRows(rowsPerBatch, uniformRows);
        if (coreNum == 0U || coreId >= coreNum || totalRows == 0U) {
            blockCnt = 0;
            batchId = batchSize;
            return;
        }
        // When Q has one inner block, every outer K row has the same fixed
        // pipeline cost. For sufficiently many rows, a direct row split avoids
        // rebuilding the full weighted prefix on every core. Mask predictors
        // still decide visibility, so this changes ownership only.
        if (useUniformOuterBlockFastPath && !usePackedSparseWorkload &&
            totalRows >= static_cast<uint64_t>(coreNum) * 4U) {
            const auto range = hstu::schedule::UniformCoreRange(totalRows, coreId, coreNum);
            blockCnt = range.endBlock - range.beginBlock;
            if (blockCnt != 0U) {
                if (uniformRows) InitUniformBlock(range.beginBlock, rowsPerBatch);
                else InitBlock(range.beginBlock);
            }
            else batchId = batchSize;
            return;
        }
        const bool plainSingleBatch = batchSize == 1U && !useTargetWorkload &&
            !useCausalWorkload && !usePackedSparseWorkload;
        if (plainSingleBatch) {
            const auto range = hstu::schedule::UniformCoreRange(totalRows, coreId, coreNum);
            blockCnt = range.endBlock - range.beginBlock;
            if (blockCnt != 0U) InitBlock(range.beginBlock);
            else batchId = batchSize;
            return;
        }

        uint64_t totalWorkload = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            const uint32_t seqLen = seqOffsetM.GetValue(batch + 1) - seqOffsetM.GetValue(batch);
            const uint32_t rows = RuntimeCeilDiv(seqLen, blockM);
            for (uint32_t head = 0; head < headNum; ++head) {
                for (uint32_t row = 0; row < rows; ++row) {
                    totalWorkload += GetRowWorkload(batch, head, row, false);
                }
            }
        }
        const bool denseFallback = usePackedSparseWorkload && totalWorkload == 0U && batchSize != 1U;
        if (denseFallback) {
            for (uint32_t batch = 0; batch < batchSize; ++batch) {
                const uint32_t seqLen = seqOffsetM.GetValue(batch + 1) - seqOffsetM.GetValue(batch);
                const uint32_t rows = RuntimeCeilDiv(seqLen, blockM);
                for (uint32_t head = 0; head < headNum; ++head) {
                    for (uint32_t row = 0; row < rows; ++row) {
                        totalWorkload += GetRowWorkload(batch, head, row, true);
                    }
                }
            }
        }
        hstu::schedule::BoundaryPolicy policy = hstu::schedule::BoundaryPolicy::FIXED_POST_ROW;
        if ((usePackedSparseWorkload || useTargetWorkload || useCausalWorkload) && batchSize == 1U) {
            policy = hstu::schedule::BoundaryPolicy::NEAREST;
        } else if (usePackedSparseWorkload && !denseFallback) {
            policy = hstu::schedule::BoundaryPolicy::NEAREST;
        } else if (useTargetWorkload || useCausalWorkload) {
            policy = hstu::schedule::BoundaryPolicy::REMAINING_POST_ROW;
        }
        hstu::schedule::WeightedCoreRangeBuilder builder(totalRows, totalWorkload, coreId, coreNum, policy);
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            const uint32_t seqLen = seqOffsetM.GetValue(batch + 1) - seqOffsetM.GetValue(batch);
            const uint32_t rows = RuntimeCeilDiv(seqLen, blockM);
            for (uint32_t head = 0; head < headNum; ++head) {
                for (uint32_t row = 0; row < rows; ++row) {
                    builder.Push(GetRowWorkload(batch, head, row, denseFallback));
                }
            }
        }
        const auto range = builder.Finish();
        blockCnt = range.endBlock - range.beginBlock;
        if (blockCnt != 0U) InitBlock(range.beginBlock);
        else batchId = batchSize;
    }

    CATLASS_DEVICE
    void Update()
    {
        if (batchId >= batchSize) {
            blockSize = 0;
            blockOffset = 0;
            return;
        }

        auto seqOffset = headBlockId * blockM;
        blockOffset = (batchBaseOffset + seqOffset) * headNum + headId;
        auto remainingSeq = currentSeqLen - seqOffset;
        blockSize = (remainingSeq < blockM) ? remainingSeq : blockM;
    }

    uint32_t blockSize{0};
    uint32_t blockOffset{0};
    uint32_t batchId{0};
    uint32_t batchSize{0};
    uint32_t batchBaseOffset{0};
    uint32_t headId{0};
    uint32_t headBlockId{0};
    uint32_t headNum{0};
    uint32_t headBlockCnt{0};
    uint32_t currentSeqLen{0};
    uint32_t blockCnt{0};
    bool useTargetWorkload{false};
    bool useCausalWorkload{false};
    bool usePackedSparseWorkload{false};
    bool useSingleOuterBlockFastPath{false};
    bool useUniformOuterBlockFastPath{false};
    uint32_t uniformOuterSeqLen{0U};
    bool hasContext{false};
    uint32_t maskHeadNum{0};
    uint32_t sparseOuterStride{0};
    uint32_t targetGroupSize{0};
    AscendC::GlobalTensor<ElementOffset> numContext;
    AscendC::GlobalTensor<ElementOffset> numTarget;
    AscendC::GlobalTensor<ElementOffset> seqOffsetM;
    AscendC::GlobalTensor<ElementOffset> seqOffsetN;
    AscendC::GlobalTensor<int32_t> maskBlockCnt;
    AscendC::GlobalTensor<int32_t> fullBlockCnt;
    uint32_t blockM{BLOCK_M};
    uint32_t blockN{BLOCK_N};
};

/**
 • @brief 列方向块调度器

 • @tparam OuterBlockScheduler_ 行方向调度器类型

 • @tparam BLOCK_M 行方向的块大小

 • @tparam BLOCK_N 列方向的块大小

 • @tparam USE_SWIZZLE 是否启用 Swizzle 优化

 • @description 负责在列方向（seq 维度）上进行任务分片，支持双向遍历 (正序/逆序)，

 •              可选支持 Swizzle 优化以提高缓存命中率

 */
template <class OuterBlockScheduler_, uint32_t BLOCK_M, uint32_t BLOCK_N, bool USE_SWIZZLE = false,
          bool PER_CORE_REVERSE = false, uint32_t KV_CLUSTER_NUM = 1, bool SPLIT_RUNTIME_INNER = false>
class InnerBlockScheduler {
public:
    using OuterBlockScheduler = OuterBlockScheduler_;
    using ElementOffset = typename OuterBlockScheduler::ElementOffset;

    /**
     ◦ @brief 构造函数

     ◦ @param batchSize 批次大小

     ◦ @param headNum 注意力头数

     ◦ @param seqOffset 序列偏移量指针

     ◦ @description 初始化列方向调度器

     */
    CATLASS_DEVICE
    InnerBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffset,
                         uint32_t runtimeBlockM = BLOCK_M, uint32_t runtimeBlockN = BLOCK_N)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->blockM = runtimeBlockM;
        this->blockN = runtimeBlockN;
        gSeqOffset.SetGlobalBuffer((__gm__ ElementOffset*)seqOffset);
    }

    CATLASS_DEVICE void EnablePackedSparse(GM_ADDR sparseInfo, uint32_t maskHeadNum, uint32_t maxOuterBlocks,
                                           uint32_t maxInnerBlocks)
    {
        if (sparseInfo == nullptr || maskHeadNum == 0) {
            return;
        }
        AscendC::ListTensorDesc sparseListDesc;
        sparseListDesc.Init(sparseInfo);
        GM_ADDR statePtr = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(6);
        packedState.SetGlobalBuffer((__gm__ int32_t*)statePtr);
        this->maskHeadNum = maskHeadNum;
        this->maxOuterBlocks = maxOuterBlocks;
        this->maxInnerBlocks = maxInnerBlocks;
        usePackedSparse = true;
    }

    CATLASS_DEVICE void EnableCausal(GM_ADDR numContext, GM_ADDR numTarget, uint32_t targetGroupSize,
                                     bool hasContext, bool hasTarget)
    {
        useCausal = true;
        this->hasContext = hasContext;
        this->hasTarget = hasTarget;
        this->targetGroupSize = targetGroupSize;
        if (hasContext) this->numContext.SetGlobalBuffer((__gm__ ElementOffset*)numContext);
        if (hasTarget) this->numTarget.SetGlobalBuffer((__gm__ ElementOffset*)numTarget);
    }

    CATLASS_DEVICE void EnableLocal(int32_t windowSizeLeft, int32_t windowSizeRight, bool forward)
    {
        useLocal = true;
        this->windowSizeLeft = windowSizeLeft;
        this->windowSizeRight = windowSizeRight;
        localForward = forward;
    }

    /**
     ◦ @brief 初始化列调度器

     ◦ @param rowBlockScheduler 行方向调度器引用

     ◦ @description 根据行调度器的元数据初始化列调度器，包括 batchId、headId、序列长度等

     ◦              如果启用 Swizzle，还可以切换遍历方向

     */
    CATLASS_DEVICE
    void Init(OuterBlockScheduler const& rowBlockScheduler)
    {
        auto meta = rowBlockScheduler.GetMeta();
        auto batchId = tla::get<0>(meta);  // 0 means batchId
        auto headId = tla::get<1>(meta);   // 1 means headId
        rowBlockId = tla::get<2>(meta);    // 2 means rowBlockId

        triggerSwizzle = false;
        swizzlePosition = 0;

        if constexpr (USE_SWIZZLE) {
            if (isInitialized && this->batchId == batchId && this->headId == headId) {
                swizzleDir = 1 - swizzleDir;
                triggerSwizzle = true;
            } else {
                swizzleDir = 1;
                isInitialized = true;
            }
        } else {
            swizzleDir = 1;
        }

        this->batchId = batchId;
        this->headId = headId;
        this->blockBaseOffset = gSeqOffset.GetValue(batchId);
        this->seqLens = gSeqOffset.GetValue(batchId + 1) - gSeqOffset.GetValue(batchId);
        this->physicalBlockCnt = (seqLens + blockN - 1) / blockN;
        this->blockBegin = 0;
        this->blockCnt = physicalBlockCnt;
        if (useCausal) {
            const uint32_t seqLenQ = rowBlockScheduler.GetCurrentSeqLen();
            const uint32_t qBegin = rowBlockId * blockM;
            const uint32_t qEnd = min(qBegin + blockM, seqLenQ) - 1;
            const uint32_t targetNum = hasTarget ? static_cast<uint32_t>(numTarget.GetValue(batchId)) : 0U;
            const uint32_t contextNum = hasContext ? static_cast<uint32_t>(numContext.GetValue(batchId)) : 0U;
            int64_t visibleEnd = static_cast<int64_t>(qEnd) + static_cast<int64_t>(seqLens) - seqLenQ + 1;
            if (hasContext && qBegin < contextNum) {
                visibleEnd = max(visibleEnd, static_cast<int64_t>(seqLens - targetNum));
            }
            if (hasTarget && targetGroupSize > 0 && qEnd >= seqLenQ - targetNum) {
                const uint32_t targetRow = qEnd - (seqLenQ - targetNum);
                const uint32_t groupEnd = seqLens - targetNum +
                    (targetRow / targetGroupSize + 1) * targetGroupSize;
                visibleEnd = max(visibleEnd, static_cast<int64_t>(min(groupEnd, seqLens)));
            }
            const uint32_t visible = visibleEnd <= 0 ? 0U : min(static_cast<uint32_t>(visibleEnd), seqLens);
            this->blockCnt = RuntimeCeilDiv(visible, blockN);
        }
        if (useLocal) {
            const uint32_t rowSeqLen = rowBlockScheduler.GetCurrentSeqLen();
            const int64_t delta = localForward ? static_cast<int64_t>(seqLens) - rowSeqLen
                                               : static_cast<int64_t>(rowSeqLen) - seqLens;
            const int64_t rowBegin = static_cast<int64_t>(rowBlockId) * blockM;
            const int64_t rowEnd = min(rowBegin + blockM, static_cast<int64_t>(rowSeqLen)) - 1;
            int64_t colBegin;
            int64_t colEnd;
            if (localForward) {
                colBegin = windowSizeLeft < 0 ? 0 : rowBegin + delta - windowSizeLeft;
                colEnd = windowSizeRight < 0 ? seqLens - 1 : rowEnd + delta + windowSizeRight;
            } else {
                colBegin = windowSizeRight < 0 ? 0 : rowBegin - delta - windowSizeRight;
                colEnd = windowSizeLeft < 0 ? seqLens - 1 : rowEnd - delta + windowSizeLeft;
            }
            colBegin = colBegin < 0 ? 0 : colBegin;
            const int64_t sequenceEnd = static_cast<int64_t>(seqLens) - 1;
            colEnd = colEnd > sequenceEnd ? sequenceEnd : colEnd;
            if (colEnd < colBegin) {
                blockBegin = 0;
                blockCnt = 0;
            } else {
                blockBegin = static_cast<uint32_t>(colBegin / blockN);
                blockCnt = static_cast<uint32_t>(colEnd / blockN) + 1;
            }
        }
        this->blockId = (swizzleDir == 1) ? blockBegin : blockCnt - 1;
        subBlockId = 0U;
        sparseFirst = true;
        if (usePackedSparse) {
            this->blockId = FindNextNonEmpty(this->blockId);
        }
        Update();
    }

    /**
     ◦ @brief 获取当前遍历是否发生 Swizzle 折返

     ◦ @return bool 当前遍历是否与同一
     * batch/head 的上一次遍历方向相反

     */
    CATLASS_DEVICE
    bool GetTriggerSwizzle() const
    {
        return triggerSwizzle;
    }

    /**
     ◦ @brief 获取当前 block 在本轮 Swizzle 遍历中的位置

     ◦ @return uint32_t 从 0
     * 开始的遍历位置，该值不是 blockId

     */
    CATLASS_DEVICE
    uint32_t GetSwizzlePosition() const
    {
        return swizzlePosition;
    }

    /**
     ◦ @brief 判断是否是第一个块

     ◦ @return bool 根据遍历方向判断是否是起始块

     */
    CATLASS_DEVICE
    bool IsFirst()
    {
        if (usePackedSparse) {
            return sparseFirst && subBlockId == 0U;
        }
        const bool firstLogical = (swizzleDir == 1) ? blockId == static_cast<int32_t>(blockBegin)
                                                    : blockId == (blockCnt - 1);
        return firstLogical && subBlockId == 0U;
    }

    /**
     ◦ @brief 判断是否是最后一个块

     ◦ @return bool 根据遍历方向判断是否是结束块

     */
    CATLASS_DEVICE
    bool IsLast()
    {
        if (subBlockId + 1U < subBlockCount) return false;
        if (usePackedSparse) {
            const int32_t candidate = (swizzleDir == 1) ? blockId + 1 : blockId - 1;
            return FindNextNonEmpty(candidate) < 0;
        }
        return (swizzleDir == 1) ? blockId == (blockCnt - 1) : blockId == 0;
    }

    /**
     ◦ @brief 判断块调度是否有效

     ◦ @return bool 根据遍历方向判断块 ID 是否在有效范围内

     */
    CATLASS_DEVICE
    bool IsValid()
    {
        if (usePackedSparse) {
            return blockId >= 0 && blockId < static_cast<int32_t>(blockCnt);
        }
        return !((swizzleDir == 1) ? blockId >= blockCnt : blockId < static_cast<int32_t>(blockBegin));
    }

    CATLASS_DEVICE int32_t GetBlockId() const
    {
        return blockId;
    }
    CATLASS_DEVICE uint32_t GetCacheBlockId() const
    {
        const uint32_t chunksPerLogical = SPLIT_RUNTIME_INNER ? RuntimeCeilDiv(blockN, BLOCK_N) : 1U;
        return static_cast<uint32_t>(blockId) * chunksPerLogical + subBlockId;
    }
    CATLASS_DEVICE bool IsFirstSubBlock() const { return subBlockId == 0U; }
    CATLASS_DEVICE bool IsLastSubBlock() const { return subBlockId + 1U == subBlockCount; }
    CATLASS_DEVICE uint32_t GetRowBlockId() const
    {
        return rowBlockId;
    }
    CATLASS_DEVICE uint32_t GetSeqLens() const
    {
        return seqLens;
    }
    CATLASS_DEVICE uint32_t GetSwizzleDir() const
    {
        return swizzleDir;
    }

    /**
     ◦ @brief 前缀递增运算符，移动到下一个块

     ◦ @return InnerBlockScheduler& 调度器引用

     ◦ @description 根据遍历方向 (swizzleDir) 更新 blockId，移动到下一个列块

     */
    CATLASS_DEVICE
    InnerBlockScheduler& operator++()
    {
        if (subBlockId + 1U < subBlockCount) {
            ++subBlockId;
            ++swizzlePosition;
            Update();
            return *this;
        }
        if (usePackedSparse) {
            const int32_t oldBlockId = blockId;
            const int32_t candidate = (swizzleDir == 1) ? blockId + 1 : blockId - 1;
            blockId = FindNextNonEmpty(candidate);
            sparseFirst = false;
            subBlockId = 0U;
            if (blockId >= 0) {
                swizzlePosition += static_cast<uint32_t>(blockId > oldBlockId ? blockId - oldBlockId
                                                                            : oldBlockId - blockId);
            }
            Update();
            return *this;
        }
        ++swizzlePosition;
        blockId = (swizzleDir == 1) ? blockId + 1 : blockId - 1;
        subBlockId = 0U;
        Update();
        return *this;
    }

    CATLASS_DEVICE void SkipBlocks(uint32_t count)
    {
        swizzlePosition += count;
        blockId = (swizzleDir == 1) ? blockId + static_cast<int32_t>(count)
                                    : blockId - static_cast<int32_t>(count);
        Update();
    }

    /**
     ◦ @brief 获取当前块的张量切片

     ◦ @tparam Tensor 张量类型

     ◦ @param tensor 源张量

     ◦ @return TensorTile 从源张量中切取的块

     ◦ @description 根据当前 blockId、blockOffset、blockSize 等信息，从张量中切取对应的数据块

     */
    template <class Tensor>
    CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(blockOffset + headId, 0);
        auto shape = tla::MakeShape(blockSize, dim);

        return tla::GetTile(tensor, coord, shape);
    }

    /**
     ◦ @brief 获取共享内存块的张量切片

     ◦ @tparam Tensor 张量类型

     ◦ @param tensor 源张量

     ◦ @param totalSeqLens 总序列长度

     ◦ @return TensorTile 共享内存块的张量切片

     ◦ @description 用于获取共享内存中的数据块，坐标计算考虑了 headId 和总序列长度

     */
    template <class Tensor>
    CATLASS_DEVICE auto GetShareTile(Tensor const& tensor, int64_t totalSeqLens)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(headId * totalSeqLens + blockOffset / headNum, 0);
        auto shape = tla::MakeShape(blockSize, dim);
        return tla::GetTile(tensor, coord, shape);
    }

    /**
     ◦ @brief 获取张量切片的映射信息

     ◦ @tparam Coord 坐标类型

     ◦ @tparam Shape 形状类型

     ◦ @param kCoord 关键坐标

     ◦ @param kShape 关键形状

     ◦ @return Tuple<Coord, Shape> 坐标和形状的元组

     ◦ @description 生成用于定位张量数据的坐标和形状信息，用于多核数据分发

     */
    template <class Coord, class Shape>
    CATLASS_DEVICE auto GetTileMapping(Coord const& kCoord, Shape const& kShape)
    {
        const uint32_t physicalN = SPLIT_RUNTIME_INNER ? BLOCK_N : blockN;
        auto coord = tla::MakeCoord(batchId, headId,
                                    blockId * blockN + subBlockId * physicalN, rowBlockId * blockM);
        auto shape = tla::MakeShape(blockSize, tla::get<0>(kShape));
        return tla::MakeTuple(coord, shape);
    }

private:
    CATLASS_DEVICE bool IsPackedNonEmpty(uint32_t innerBlock) const
    {
        const uint32_t descriptor =
            static_cast<uint32_t>(packedState.GetValue(GetPackedOuter() * maxInnerBlocks + innerBlock));
        const uint32_t state = descriptor & 0x3U;
        return state == 0x1U || state == 0x3U;
    }

    CATLASS_DEVICE uint32_t GetPackedOuter() const
    {
        const uint32_t maskHead = maskHeadNum == 1 ? 0 : headId;
        return (batchId * maskHeadNum + maskHead) * maxOuterBlocks + rowBlockId;
    }

    CATLASS_DEVICE int32_t FindNextNonEmpty(int32_t candidate) const
    {
        if (swizzleDir == 1) {
            for (; candidate < static_cast<int32_t>(blockCnt); ++candidate) {
                if (IsPackedNonEmpty(static_cast<uint32_t>(candidate))) return candidate;
            }
        } else {
            for (; candidate >= 0; --candidate) {
                if (IsPackedNonEmpty(static_cast<uint32_t>(candidate))) return candidate;
            }
        }
        return -1;
    }

    /**
     ◦ @brief 更新块信息

     ◦ @description 根据当前 blockId 计算 blockSize 和 blockOffset

     */
    CATLASS_DEVICE
    void Update()
    {
        if (!IsValid()) {
            blockSize = 0;
            blockOffset = 0;
            return;
        }
        const uint32_t logicalSize = (blockId == (physicalBlockCnt - 1)) ?
            (seqLens - blockId * blockN) : blockN;
        const uint32_t physicalN = SPLIT_RUNTIME_INNER ? BLOCK_N : blockN;
        subBlockCount = RuntimeCeilDiv(logicalSize, physicalN);
        const uint32_t localOffset = subBlockId * physicalN;
        blockSize = min(physicalN, logicalSize - localOffset);
        blockOffset = (blockBaseOffset + blockId * blockN + localOffset) * headNum;
    }

    uint32_t blockSize{0};
    uint32_t subBlockId{0};
    uint32_t subBlockCount{1};
    uint32_t blockOffset{0};
    uint32_t blockBaseOffset{0};
    uint32_t batchSize{0};
    uint32_t headNum{0};
    uint32_t batchId{0};
    uint32_t headId{0};
    uint32_t seqLens{0};
    int32_t blockId{0};
    uint32_t rowBlockId{0};
    uint32_t blockCnt{0};
    uint32_t blockBegin{0};
    uint32_t physicalBlockCnt{0};
    bool isInitialized{false};
    // 当前遍历是否相对同一 batch/head 的上一次遍历发生折返；为 true 时才允许尝试复用 L1 数据。
    bool triggerSwizzle{false};
    bool usePackedSparse{false};
    bool useCausal{false};
    bool useLocal{false};
    bool localForward{true};
    bool hasContext{false};
    bool hasTarget{false};
    bool sparseFirst{true};
    uint32_t maskHeadNum{0};
    uint32_t maxOuterBlocks{0};
    uint32_t maxInnerBlocks{0};
    uint32_t targetGroupSize{0};
    int32_t windowSizeLeft{-1};
    int32_t windowSizeRight{-1};
    // 当前 block 在本轮遍历中的序号；Init() 时清零，operator++() 时递增。
    uint32_t swizzlePosition{0};
    uint32_t swizzleDir{1};
    AscendC::GlobalTensor<ElementOffset> gSeqOffset;
    uint32_t blockM{BLOCK_M};
    uint32_t blockN{BLOCK_N};
    AscendC::GlobalTensor<int32_t> packedState;
    AscendC::GlobalTensor<ElementOffset> numContext;
    AscendC::GlobalTensor<ElementOffset> numTarget;
};

}  // namespace Catlass::Gemm::Block
