/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/library.h>
#include "../common/attn_tile_size.h"

namespace ops_rec::attn_plan {

using TensorList = std::vector<at::Tensor>;
using DirectionPlanResult = std::tuple<std::vector<int64_t>, TensorList, c10::optional<at::Tensor>, bool>;
enum class Direction : int64_t {
    FORWARD = 0,
    BACKWARD = 1
};
using RuntimeTileStrategy = TilePlan (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                         const CoreMemSize&, bool);
using AicpuSchedulePolicy = bool (*)(uint64_t, uint64_t);

uint64_t GetRuntimeL2Size();

struct DirectionPlan {
    std::vector<int64_t> tile;
    TensorList sparseInfo;
    c10::optional<at::Tensor> metadata;
    bool useAicpuSchedule;
};

DirectionPlan MakeDirectionPlan(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const c10::optional<at::Tensor>& af, const at::Tensor& seqOffsetQ,
                                const at::Tensor& seqOffsetK, int64_t maxSeqLenQ, int64_t maxSeqLenK,
                                Direction direction, const c10::optional<at::Tensor>& numContext,
                                const c10::optional<at::Tensor>& numTarget, int64_t targetGroupSize,
                                int64_t windowSizeLeft, int64_t windowSizeRight, int64_t maskType, bool deterministic,
                                RuntimeTileStrategy tileStrategy, AicpuSchedulePolicy schedulePolicy);

DirectionPlanResult GetFwdAttnPlan(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   const c10::optional<at::Tensor>& af, const at::Tensor& seqOffsetQ,
                                   const at::Tensor& seqOffsetK, int64_t maxSeqLenQ, int64_t maxSeqLenK,
                                   const c10::optional<at::Tensor>& numContext,
                                   const c10::optional<at::Tensor>& numTarget, int64_t targetGroupSize,
                                   int64_t windowSizeLeft, int64_t windowSizeRight, int64_t maskType,
                                   bool deterministic = true);

DirectionPlanResult GetBwdAttnPlan(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   const c10::optional<at::Tensor>& af, const at::Tensor& seqOffsetQ,
                                   const at::Tensor& seqOffsetK, int64_t maxSeqLenQ, int64_t maxSeqLenK,
                                   const c10::optional<at::Tensor>& numContext,
                                   const c10::optional<at::Tensor>& numTarget, int64_t targetGroupSize,
                                   int64_t windowSizeLeft, int64_t windowSizeRight, int64_t maskType,
                                   bool deterministic = true);

}  // namespace ops_rec::attn_plan
