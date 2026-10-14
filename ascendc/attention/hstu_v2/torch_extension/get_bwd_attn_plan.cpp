/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include "../../attn_plan/host/attn_plan.h"
#include "../common/hstu_tile_size.h"

namespace {
bool NeverUseAicpu(uint64_t, uint64_t) { return false; }
}

namespace ops_rec::attn_plan {
DirectionPlanResult GetBwdAttnPlan(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const c10::optional<at::Tensor>& af,
    const at::Tensor& seqOffsetQ, const at::Tensor& seqOffsetK, int64_t maxSeqLenQ, int64_t maxSeqLenK,
    const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget, int64_t targetGroupSize,
    int64_t windowSizeLeft, int64_t windowSizeRight, int64_t maskType, bool deterministic)
{
    auto plan = MakeDirectionPlan(q, k, v, af, seqOffsetQ, seqOffsetK, maxSeqLenQ, maxSeqLenK, Direction::BACKWARD,
                                  numContext, numTarget, targetGroupSize, windowSizeLeft, windowSizeRight, maskType,
                                  deterministic, &ops_rec::hstu::GetBwdRuntimeTile, &NeverUseAicpu);
    return {std::move(plan.tile), std::move(plan.sparseInfo), std::move(plan.metadata), plan.useAicpuSchedule};
}
}  // namespace ops_rec::attn_plan
