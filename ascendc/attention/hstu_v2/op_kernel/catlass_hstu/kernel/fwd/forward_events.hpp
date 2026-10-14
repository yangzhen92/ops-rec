/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include "kernel_operator.h"

namespace Catlass::Kernel {

struct ForwardMmadEvents {
    static constexpr uint32_t K_L1_0 = EVENT_ID0, K_L1_1 = EVENT_ID1;
    static constexpr uint32_t Q_L1_0 = EVENT_ID6, Q_L1_1 = EVENT_ID7;
    static constexpr uint32_t V_L1_0 = EVENT_ID4, V_L1_1 = EVENT_ID5;
    static constexpr uint32_t QK_L0_HANDOFF = EVENT_ID2, PV_L0_HANDOFF = EVENT_ID3;
    static constexpr uint32_t QK_L0C_FIX = EVENT_ID4, PV_L0C_FIX = EVENT_ID5;
    static constexpr uint32_t QK_READY_0 = EVENT_ID0, QK_READY_1 = EVENT_ID1, QK_READY_2 = EVENT_ID2;
    static constexpr uint32_t PV_READY_0 = EVENT_ID3, PV_READY_1 = EVENT_ID4, PV_READY_2 = EVENT_ID5;
    static constexpr uint32_t TRANS_READY = EVENT_ID6;
};

struct ForwardEpilogueEvents {
    static constexpr uint32_t QK_READY_0 = EVENT_ID0, QK_READY_1 = EVENT_ID1, QK_READY_2 = EVENT_ID2;
    static constexpr uint32_t PV_READY_0 = EVENT_ID3, PV_READY_1 = EVENT_ID4, PV_READY_2 = EVENT_ID5;
    static constexpr uint32_t TRANS_READY = EVENT_ID6;
    static constexpr uint32_t TRANS_MTE3 = EVENT_ID0;
};

static_assert(ForwardMmadEvents::QK_READY_2 < ForwardMmadEvents::PV_READY_0);
static_assert(ForwardMmadEvents::PV_READY_2 < ForwardMmadEvents::TRANS_READY);

}  // namespace Catlass::Kernel
