/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>
#include "../../../common/hstu_tile_size.h"
#include "../../tla_hstu/layout.hpp"

namespace Catlass::Kernel {
enum class HstuDirection : uint8_t { FORWARD, BACKWARD };

// Forward and backward both use one Cube core with two Vector sub-cores.
// Backward SAME AB splits each dV/dK result along M across the two AIVs.
#define HSTU_V2_TASK_TYPE KERNEL_TYPE_MIX_AIC_1_2

template <HstuDirection Direction, uint32_t HeadDim>
struct HstuPipelinePolicy {
    static_assert(HeadDim == 128U || HeadDim == 256U, "Unsupported HSTU head dimension tile");
};

template <>
struct HstuPipelinePolicy<HstuDirection::FORWARD, 128U> {
    static constexpr auto TILE = ops_rec::hstu::GetFwdTileSize(128U);
    using L1TileShape = tla::Shape<tla::Int<TILE.m>, tla::Int<TILE.n>, tla::Int<TILE.k>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    static constexpr uint32_t AIV_PER_AIC = 2U;
    static constexpr bool SUPPORTS_L0_SPLIT_K = false;
};
template <>
struct HstuPipelinePolicy<HstuDirection::FORWARD, 256U> {
    static constexpr auto TILE = ops_rec::hstu::GetFwdTileSize(256U);
    using L1TileShape = tla::Shape<tla::Int<TILE.m>, tla::Int<TILE.n>, tla::Int<TILE.k>>;
    using L0TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<256>>;
    static constexpr uint32_t AIV_PER_AIC = 2U;
    static constexpr bool SUPPORTS_L0_SPLIT_K = false;
};
template <>
struct HstuPipelinePolicy<HstuDirection::BACKWARD, 128U> {
    static constexpr auto TILE = ops_rec::hstu::GetBwdTileSize(128U);
    using L1TileShape = tla::Shape<tla::Int<TILE.m>, tla::Int<TILE.n>, tla::Int<TILE.k>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    static constexpr uint32_t AIV_PER_AIC = 2U;
    static constexpr bool SUPPORTS_L0_SPLIT_K = false;
};
template <>
struct HstuPipelinePolicy<HstuDirection::BACKWARD, 256U> {
    static constexpr auto TILE = ops_rec::hstu::GetBwdTileSize(256U);
    using L1TileShape = tla::Shape<tla::Int<TILE.m>, tla::Int<TILE.n>, tla::Int<TILE.k>>;
    using L0TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<256>>;
    static constexpr uint32_t AIV_PER_AIC = 2U;
    static constexpr bool SUPPORTS_L0_SPLIT_K = false;
};
}  // namespace Catlass::Kernel
