/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>

namespace ops_rec::attn_plan::metadata {
constexpr uint32_t HEAD_STRIDE = 16U;
constexpr uint32_t FA_STRIDE = 16U;
constexpr uint32_t FD_STRIDE = 16U;
constexpr uint32_t HEAD_SECTION_NUM = 0U;
constexpr uint32_t HEAD_IS_FD = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE = 3U;
constexpr uint32_t HEAD_AIV_CORE_NUM = 13U;
constexpr uint32_t HEAD_AIC_CORE_NUM = 14U;
constexpr uint32_t HEAD_DIRECT_MAGIC = 15U;
// Direct dense metadata:
//   int32 direct_task[aicCoreNum][HEAD_STRIDE]
// where each record uses [DIRECT_TASK_BEGIN, DIRECT_TASK_COUNT].
constexpr uint32_t DIRECT_MAGIC = 0x48535455U;
// Stream-K metadata keeps one [global tile begin, tile count] record per
// section/core.  Tile ids use the same batch/head/outer/inner order as
// StreamKOuterBlockScheduler.
constexpr uint32_t STREAM_K_MAGIC = 0x534B4855U;
constexpr uint32_t DIRECT_TASK_BEGIN = 0U;
constexpr uint32_t DIRECT_TASK_COUNT = 1U;
constexpr uint32_t FA_BN2_START = 0U;
constexpr uint32_t FA_M_START = 1U;
constexpr uint32_t FA_S2_START = 2U;
constexpr uint32_t FA_BN2_END = 3U;
constexpr uint32_t FA_M_END = 4U;
constexpr uint32_t FA_S2_END = 5U;
constexpr uint32_t FA_FIRST_FD_WORKSPACE = 6U;
constexpr uint32_t FD_BN2 = 0U;
constexpr uint32_t FD_M = 1U;
constexpr uint32_t FD_WORKSPACE = 2U;
constexpr uint32_t FD_WORKSPACE_NUM = 3U;
constexpr uint32_t FD_M_START = 4U;
constexpr uint32_t FD_M_NUM = 5U;
}  // namespace ops_rec::attn_plan::metadata
