/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>
#include "../../attn_plan/common/attn_tile_size.h"

namespace ops_rec::hstu {

struct TileSize {
    uint32_t m;  // Q sequence axis
    uint32_t n;  // K/V sequence axis
    uint32_t k;  // head-dimension tile
};
using CoreMemSize = ops_rec::attn_plan::CoreMemSize;
using TilePlan = ops_rec::attn_plan::TilePlan;

struct HstuTileTrait {
    static constexpr TilePlan GetFwdPhysicalCapacity(uint32_t headDim)
    {
        return headDim <= 128U ? TilePlan{{128U, 128U, 128U}, {128U, 128U, 128U}}
                               : TilePlan{{64U, 64U, 256U}, {64U, 64U, 256U}};
    }

    static constexpr TilePlan GetBwdPhysicalCapacity(uint32_t headDim)
    {
        return headDim <= 128U ? TilePlan{{256U, 128U, 128U}, {128U, 128U, 128U}}
                               : TilePlan{{128U, 64U, 256U}, {64U, 64U, 256U}};
    }

    static bool CheckFwdBuffer(uint32_t m, uint32_t n, uint32_t dimQk, uint32_t dimV, const CoreMemSize& mem,
                               const TilePlan& physical)
    {
        if (m > physical.l1.m || dimQk > physical.l1.k || dimV > physical.l1.k)
            return false;
        const auto l0 = ops_rec::attn_plan::GetL0Tile(m, n, dimQk, dimV, mem, physical.l0);
        if (l0.m == 0U || l0.n == 0U || l0.k == 0U)
            return false;
        constexpr uint64_t ELEMENT_BYTES = 2U;
        constexpr uint64_t ACC_BYTES = 4U;
        const uint64_t qkL0a = static_cast<uint64_t>(m) * dimQk * ELEMENT_BYTES;
        const uint64_t pvL0a = static_cast<uint64_t>(m) * n * ELEMENT_BYTES;
        const uint64_t qkL0b = static_cast<uint64_t>(n) * dimQk * ELEMENT_BYTES;
        const uint64_t pvL0b = static_cast<uint64_t>(n) * dimV * ELEMENT_BYTES;
        const uint64_t qkL0c = static_cast<uint64_t>(m) * n * ACC_BYTES;
        const uint64_t pvL0c = static_cast<uint64_t>(m) * dimV * ACC_BYTES;
        // Q/K/V keep three pipeline slots in the shared L1 layout.  K and V
        // are resident for the complete runtime N tile; score/prob remain
        // physical L0 sub-tiles and are consumed a slice at a time.
        const uint64_t blockK = std::max(dimQk, dimV);
        const uint64_t l1 = (2ULL * m * blockK + 4ULL * n * blockK + 3ULL * m * n) * ELEMENT_BYTES;
        constexpr uint64_t AIV_PER_AIC = 2U;
        const uint64_t score = static_cast<uint64_t>(m) * std::max(n, static_cast<uint32_t>(blockK)) / AIV_PER_AIC;
        const uint64_t prob = static_cast<uint64_t>(m) * n / AIV_PER_AIC;
        const uint64_t output = static_cast<uint64_t>(m) * blockK / AIV_PER_AIC;
        const uint64_t packedMask = static_cast<uint64_t>(m) / AIV_PER_AIC * ops_rec::attn_plan::CeilDiv(n, 64U) * 32U;
        const uint64_t ub = score * 2U * ELEMENT_BYTES + prob * 3U * ACC_BYTES + prob * 2U * ELEMENT_BYTES +
                            output * ELEMENT_BYTES + output * 2U * ACC_BYTES + packedMask;
        return qkL0a + pvL0a <= mem.l0aBytes && qkL0b + pvL0b <= mem.l0bBytes && qkL0c + pvL0c <= mem.l0cBytes &&
               l1 <= mem.l1Bytes && ub <= mem.ubBytes;
    }

    static bool CheckBwdBuffer(uint32_t m, uint32_t n, uint32_t dimQk, uint32_t dimV, const CoreMemSize& mem,
                               const TilePlan& physical)
    {
        if (n > physical.l1.n || dimQk > physical.l1.k || dimV > physical.l1.k)
            return false;
        const auto l0 = ops_rec::attn_plan::GetL0Tile(m, n, dimQk, dimV, mem, physical.l0);
        if (l0.m == 0U || l0.n == 0U || l0.k == 0U)
            return false;
        constexpr uint64_t E = 2U, A = 4U;
        const uint64_t mn = static_cast<uint64_t>(physical.l0.m) * physical.l0.n;
        const uint64_t nk = static_cast<uint64_t>(physical.l0.n) * physical.l0.k;
        const uint64_t mk = static_cast<uint64_t>(physical.l0.m) * physical.l0.k;
        const uint64_t l0c = mem.l0cBytes / 2U + 2ULL * nk * A;
        // The backward inner Q axis is streamed in physical L0-M slices, so
        // enlarging runtime M does not increase resident L1 storage. K/V use
        // the runtime outer N extent; Q/grad/prob/grab retain physical slices.
        const uint64_t l1 = (2ULL * n * std::max(dimQk, dimV) + 4ULL * mk + 4ULL * mn) * E;
        const uint64_t scoreEnd = mn * A;
        const uint64_t transOutEnd = scoreEnd + nk / 2U * (A + E);
        const uint64_t probEnd = scoreEnd + mn * E;
        const uint64_t grabEnd = std::max(transOutEnd, probEnd) + mn * E;
        const uint64_t grabPartEnd = grabEnd + mn * E;
        const uint64_t gsEnd = grabPartEnd + mn * A;
        const uint64_t packedMaskEnd = gsEnd + physical.l0.m * 64U;
        const uint64_t transKEnd = (physical.l0.k == 128U ? grabEnd : packedMaskEnd) + nk / 2U * A;
        return mem.l0aBytes <= mem.l0aBytes && mem.l0bBytes <= mem.l0bBytes && l0c <= mem.l0cBytes &&
               l1 <= mem.l1Bytes && std::max(packedMaskEnd, transKEnd) <= mem.ubBytes;
    }
};

constexpr TileSize GetFwdTileSize(uint32_t headDim)
{
    return headDim <= 128U ? TileSize{128U, 128U, 128U} : TileSize{64U, 64U, 256U};
}

constexpr TileSize GetBwdTileSize(uint32_t headDim)
{
    // Backward schedules K rows in the outer loop (N) and Q columns in
    // the inner loop (M), but M still denotes the Q-axis block size.
    return headDim <= 128U ? TileSize{256U, 128U, 128U} : TileSize{128U, 64U, 256U};
}

inline TilePlan GetFwdRuntimeTile(uint32_t seqlenQ, uint32_t seqlenK, uint32_t batch, uint32_t heads, uint32_t dimQk,
                                  uint32_t dimV, uint32_t coreNum, const CoreMemSize& mem, bool allowDynamic)
{
    return ops_rec::attn_plan::SearchFwdTileSize<HstuTileTrait>(seqlenQ, seqlenK, batch, heads, dimQk, dimV, coreNum,
                                                                mem, allowDynamic);
}

inline TilePlan GetBwdRuntimeTile(uint32_t seqlenQ, uint32_t seqlenK, uint32_t batch, uint32_t heads, uint32_t dimQk,
                                  uint32_t dimV, uint32_t coreNum, const CoreMemSize& mem, bool allowDynamic)
{
    return ops_rec::attn_plan::SearchBwdTileSize<HstuTileTrait>(seqlenQ, seqlenK, batch, heads, dimQk, dimV, coreNum,
                                                                mem, allowDynamic);
}

}  // namespace ops_rec::hstu
