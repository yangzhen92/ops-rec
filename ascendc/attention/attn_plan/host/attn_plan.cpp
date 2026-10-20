/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */

#include "attn_plan.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

#include <acl/acl_platform.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/interface/AclInterface.h>

#include "../common/attn_tile_size.h"

using at::Tensor;

ops_rec::attn_plan::TensorList create_fwd_packed_block_sparse_impl_npu(const Tensor&, const Tensor&, const Tensor&,
                                                                       int64_t, int64_t, int64_t, int64_t);
ops_rec::attn_plan::TensorList create_bwd_packed_block_sparse_impl_npu(const Tensor&, const Tensor&, const Tensor&,
                                                                       int64_t, int64_t, int64_t, int64_t);
Tensor create_attn_metadata_impl_npu(const c10::optional<Tensor>&, const c10::optional<Tensor>&,
                                     const c10::optional<Tensor>&, const c10::optional<Tensor>&, int64_t, int64_t,
                                     int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string,
                                     std::string, std::string, const c10::optional<Tensor>&,
                                     const c10::optional<Tensor>&, int64_t, int64_t, int64_t, int64_t, bool);

namespace ops_rec::attn_plan {
namespace {
bool Present(const c10::optional<Tensor>& value)
{
    return value.has_value() && value->defined();
}

uint64_t GetPlatformValue(aclplatformDevInfo info)
{
    char value[64] = {};
    const auto status = aclplatformGetDeviceInfo(info, value, sizeof(value));
    TORCH_CHECK(status == ACL_SUCCESS, "failed to query Ascend platform memory, status=", status);
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0', "invalid Ascend platform memory value: ", value);
    return parsed;
}

CoreMemSize GetCoreMemSize()
{
    return {GetPlatformValue(ACL_PLATFORM_L0_A_SIZE), GetPlatformValue(ACL_PLATFORM_L0_B_SIZE),
            GetPlatformValue(ACL_PLATFORM_L0_C_SIZE), GetPlatformValue(ACL_PLATFORM_L1_SIZE),
            GetPlatformValue(ACL_PLATFORM_AICORE_UB_SIZE)};
}
}  // namespace

uint64_t GetRuntimeL2Size()
{
    return GetPlatformValue(ACL_PLATFORM_L2_SIZE);
}

DirectionPlan MakeDirectionPlan(const Tensor& q, const Tensor& k, const Tensor& v, const c10::optional<Tensor>& af,
                                const Tensor& seqOffsetQ, const Tensor& seqOffsetK, int64_t maxSeqLenQ,
                                int64_t maxSeqLenK, Direction direction, const c10::optional<Tensor>& numContext,
                                const c10::optional<Tensor>& numTarget, int64_t targetGroupSize, int64_t windowSizeLeft,
                                int64_t windowSizeRight, int64_t maskType, bool deterministic,
                                RuntimeTileStrategy tileStrategy, AicpuSchedulePolicy schedulePolicy)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "attention plan expects TND q/k/v tensors");
    TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0,
                "q heads must be positive and divisible by kv heads");
    TORCH_CHECK(seqOffsetQ.dim() == 1 && seqOffsetK.dim() == 1 && seqOffsetQ.numel() >= 2 &&
                    seqOffsetQ.numel() == seqOffsetK.numel(),
                "sequence offsets must be one-dimensional tensors with matching length >= 2");
    TORCH_CHECK(maxSeqLenQ > 0 && maxSeqLenK > 0, "maximum sequence lengths must be positive");
    TORCH_CHECK(targetGroupSize > 0, "target_group_size must be positive");

    const int64_t batch = seqOffsetQ.numel() - 1;
    const int64_t headsQ = q.size(1);
    const int64_t headsKv = k.size(1);
    const uint32_t headDim = static_cast<uint32_t>(std::max(q.size(2), v.size(2)));
    const bool arbitrary = Present(af);
    const bool local = windowSizeLeft >= 0 || windowSizeRight > 0;
    const bool masked = arbitrary || Present(numContext) || Present(numTarget) || local || maskType != 2;
    const uint32_t coreNum =
        static_cast<uint32_t>(c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE));
    const auto mem = GetCoreMemSize();
    TORCH_CHECK(tileStrategy != nullptr && schedulePolicy != nullptr, "attention plan strategies must be provided");
    auto runtimeTile = tileStrategy(maxSeqLenQ, maxSeqLenK, batch, headsQ, q.size(2), v.size(2), coreNum, mem, true);
    const auto tile = runtimeTile.l1;
    TensorList sparseInfo;
    if (arbitrary) {
        sparseInfo = direction == Direction::FORWARD
                         ? create_fwd_packed_block_sparse_impl_npu(*af, seqOffsetQ, seqOffsetK, maxSeqLenQ, maxSeqLenK,
                                                                   tile.m, tile.n)
                         : create_bwd_packed_block_sparse_impl_npu(*af, seqOffsetQ, seqOffsetK, maxSeqLenQ, maxSeqLenK,
                                                                   tile.m, tile.n);
    }
    const uint64_t innerTokens = direction == Direction::FORWARD ? k.size(0) : q.size(0);
    const uint64_t innerHeads = direction == Direction::FORWARD ? headsKv : headsQ;
    const uint64_t dtypeBytes = q.element_size();
    TORCH_CHECK(k.element_size() == dtypeBytes && v.element_size() == dtypeBytes,
                "q/k/v must use matching element sizes");
    const uint64_t innerBytes = innerTokens * innerHeads * static_cast<uint64_t>(q.size(2) + v.size(2)) * dtypeBytes;
    const uint64_t l2Bytes = GetRuntimeL2Size();
    const bool shortSequence = maxSeqLenQ <= static_cast<int64_t>(tile.m) && maxSeqLenK <= static_cast<int64_t>(tile.n);
    const uint64_t qBlocks = (static_cast<uint64_t>(maxSeqLenQ) + tile.m - 1U) / tile.m;
    const uint64_t kBlocks = (static_cast<uint64_t>(maxSeqLenK) + tile.n - 1U) / tile.n;
    const uint64_t outerTasks =
        static_cast<uint64_t>(batch) * headsKv * (direction == Direction::FORWARD ? qBlocks : kBlocks);
    const uint64_t innerBlocks = direction == Direction::FORWARD ? kBlocks : qBlocks;
    // The existing SectionStreamK metadata can safely segment forward work
    // across complete batch/head groups.  Backward still lacks inner-Q
    // split/reduction support and measured regressions grow at 32K/64K, so
    // keep its automatic route on AICore until that protocol exists.
    // Stream-K is useful whenever it expands an under-filled outer schedule,
    // even if the complete tile grid is still smaller than the device. For
    // example, 3584x128 backward grows from two active outer cores to fourteen.
    const bool streamKCandidate = !deterministic && outerTasks < coreNum && innerBlocks > 1U;
    const bool useAicpuSchedule =
        !shortSequence &&
        ((deterministic && direction == Direction::FORWARD && schedulePolicy(innerBytes, l2Bytes)) || streamKCandidate);
    c10::optional<Tensor> metadata = c10::nullopt;
    if (useAicpuSchedule) {
        const c10::optional<Tensor> maskCount = arbitrary ? c10::optional<Tensor>(sparseInfo[0]) : c10::nullopt;
        const c10::optional<Tensor> fullCount = arbitrary ? c10::optional<Tensor>(sparseInfo[3]) : c10::nullopt;
        const int64_t packedHeads = arbitrary ? sparseInfo[0].size(1) : 1;
        const int64_t metadataMaskMode = arbitrary ? 0 : (local ? 4 : (maskType == 0 ? 3 : 0));
        metadata = create_attn_metadata_impl_npu(
            seqOffsetQ, seqOffsetK, c10::nullopt, c10::nullopt, batch, maxSeqLenQ, maxSeqLenK, headsQ, headsKv, headDim,
            metadataMaskMode, metadataMaskMode == 4 ? windowSizeLeft : -1, metadataMaskMode == 4 ? windowSizeRight : -1,
            "TND", "TND", "TND", maskCount, fullCount, static_cast<int64_t>(direction), packedHeads, tile.m, tile.n,
            deterministic);
    }
    return {{tile.m, tile.n, tile.k}, std::move(sparseInfo), std::move(metadata), useAicpuSchedule};
}

}  // namespace ops_rec::attn_plan
