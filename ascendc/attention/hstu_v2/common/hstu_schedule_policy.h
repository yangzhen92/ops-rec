/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include "../../attn_plan/common/attn_schedule_policy.h"

namespace hstu::schedule {
using ops_rec::attn_plan::schedule::BoundaryPolicy;
using ops_rec::attn_plan::schedule::CoreRange;
using ops_rec::attn_plan::schedule::L2BudgetBytes;
using ops_rec::attn_plan::schedule::ShouldUseAicpuSchedule;
using ops_rec::attn_plan::schedule::UniformBoundary;
using ops_rec::attn_plan::schedule::UniformCoreRange;
using ops_rec::attn_plan::schedule::WeightedCoreRangeBuilder;
using ops_rec::attn_plan::schedule::WeightedTarget;
}  // namespace hstu::schedule
