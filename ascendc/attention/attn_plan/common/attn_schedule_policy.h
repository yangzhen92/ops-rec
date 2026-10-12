/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>

namespace ops_rec::attn_plan::schedule {

#if defined(__CCE__)
#define ATTN_SCHEDULE_INLINE __forceinline__[host, aicore]
#else
#define ATTN_SCHEDULE_INLINE inline
#endif

ATTN_SCHEDULE_INLINE bool ShouldUseAicpuSchedule(uint64_t innerBytes, uint64_t l2Bytes, uint32_t alphaNumerator = 4U,
                                                 uint32_t alphaDenominator = 5U)
{
    if (l2Bytes == 0U || alphaNumerator == 0U || alphaDenominator == 0U)
        return false;
    const uint64_t quotient = innerBytes / l2Bytes;
    const uint64_t remainder = innerBytes % l2Bytes;
    if (quotient > alphaNumerator / alphaDenominator)
        return true;
    if (quotient < alphaNumerator / alphaDenominator)
        return false;
    return remainder > (l2Bytes / alphaDenominator) * alphaNumerator +
                           ((l2Bytes % alphaDenominator) * alphaNumerator) / alphaDenominator;
}

ATTN_SCHEDULE_INLINE uint64_t L2BudgetBytes(uint64_t l2Bytes, uint32_t alphaNumerator = 4U,
                                            uint32_t alphaDenominator = 5U)
{
    if (alphaNumerator == 0U || alphaDenominator == 0U)
        return 0U;
    return (l2Bytes / alphaDenominator) * alphaNumerator +
           ((l2Bytes % alphaDenominator) * alphaNumerator) / alphaDenominator;
}

ATTN_SCHEDULE_INLINE uint64_t WeightedTarget(uint64_t totalWeight, uint32_t partition, uint32_t partitions)
{
    return partitions == 0U ? 0U : (totalWeight * static_cast<uint64_t>(partition) + partitions - 1U) / partitions;
}

ATTN_SCHEDULE_INLINE uint32_t UniformBoundary(uint32_t totalRows, uint32_t partition, uint32_t partitions)
{
    return partitions == 0U ? 0U : static_cast<uint32_t>((static_cast<uint64_t>(totalRows) * partition) / partitions);
}

struct CoreRange {
    uint32_t beginBlock{0};
    uint32_t endBlock{0};
};
enum class BoundaryPolicy : uint8_t {
    FIXED_POST_ROW = 0,
    REMAINING_POST_ROW = 1,
    NEAREST = 2
};

ATTN_SCHEDULE_INLINE CoreRange UniformCoreRange(uint32_t totalRows, uint32_t coreId, uint32_t coreNum)
{
    if (coreNum == 0U || coreId >= coreNum)
        return {};
    const uint32_t rowsPerCore = totalRows / coreNum;
    const uint32_t remainder = totalRows % coreNum;
    const uint32_t begin = coreId * rowsPerCore + (coreId < remainder ? coreId : remainder);
    const uint32_t count = rowsPerCore + (coreId < remainder ? 1U : 0U);
    return {begin, begin + count};
}

class WeightedCoreRangeBuilder {
public:
    ATTN_SCHEDULE_INLINE WeightedCoreRangeBuilder(uint32_t totalRows, uint64_t totalWeight, uint32_t coreId,
                                                  uint32_t coreNum, BoundaryPolicy policy)
        : totalRows_(totalRows),
          remainingWeight_(totalWeight),
          requestedCore_(coreId),
          coreNum_(coreNum),
          policy_(policy)
    {
        fixedTarget_ = coreNum == 0U ? 0U : (totalWeight + coreNum - 1U) / coreNum;
        if (totalWeight == 0U) {
            range_ = UniformCoreRange(totalRows, coreId, coreNum);
            finished_ = true;
        }
    }
    ATTN_SCHEDULE_INLINE void Push(uint64_t rowWeight)
    {
        if (finished_ || currentRow_ >= totalRows_ || currentCore_ >= coreNum_)
            return;
        const uint64_t remainingCores = coreNum_ - currentCore_;
        const uint64_t target = policy_ == BoundaryPolicy::FIXED_POST_ROW
                                    ? fixedTarget_
                                    : (remainingWeight_ + remainingCores - 1U) / remainingCores;
        const uint32_t rowsNeededByLaterCores = coreNum_ - currentCore_ - 1U;
        const bool cutBefore = policy_ == BoundaryPolicy::NEAREST && currentWeight_ != 0U &&
                               currentCore_ + 1U < coreNum_ && totalRows_ - currentRow_ >= rowsNeededByLaterCores &&
                               currentWeight_ < target && currentWeight_ + rowWeight >= target &&
                               target - currentWeight_ <= currentWeight_ + rowWeight - target;
        if (cutBefore)
            Cut(currentRow_);
        currentWeight_ += rowWeight;
        ++currentRow_;
        const uint64_t updatedCores = coreNum_ - currentCore_;
        const uint64_t updatedTarget = policy_ == BoundaryPolicy::FIXED_POST_ROW
                                           ? fixedTarget_
                                           : (remainingWeight_ + updatedCores - 1U) / updatedCores;
        if (currentWeight_ >= updatedTarget && currentCore_ + 1U < coreNum_)
            Cut(currentRow_);
    }
    ATTN_SCHEDULE_INLINE CoreRange Finish()
    {
        if (!finished_ && currentCore_ == requestedCore_)
            range_ = {rowBegin_, totalRows_};
        else if (!finished_ && requestedCore_ > currentCore_ && requestedCore_ < coreNum_)
            range_ = {totalRows_, totalRows_};
        return range_;
    }

private:
    ATTN_SCHEDULE_INLINE void Cut(uint32_t rowEnd)
    {
        if (currentCore_ == requestedCore_) {
            range_ = {rowBegin_, rowEnd};
            finished_ = true;
        }
        remainingWeight_ = remainingWeight_ > currentWeight_ ? remainingWeight_ - currentWeight_ : 0U;
        currentWeight_ = 0U;
        rowBegin_ = rowEnd;
        ++currentCore_;
    }
    uint32_t totalRows_{0};
    uint64_t remainingWeight_{0};
    uint64_t currentWeight_{0};
    uint64_t fixedTarget_{0};
    uint32_t requestedCore_{0};
    uint32_t coreNum_{0};
    uint32_t currentCore_{0};
    uint32_t currentRow_{0};
    uint32_t rowBegin_{0};
    BoundaryPolicy policy_{BoundaryPolicy::REMAINING_POST_ROW};
    CoreRange range_{};
    bool finished_{false};
};

}  // namespace ops_rec::attn_plan::schedule

#undef ATTN_SCHEDULE_INLINE
