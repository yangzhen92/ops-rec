/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace ops_rec::attn_plan {

struct CoreMemSize {
    uint64_t l0aBytes{0};
    uint64_t l0bBytes{0};
    uint64_t l0cBytes{0};
    uint64_t l1Bytes{0};
    uint64_t ubBytes{0};
};

struct TileSize {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
struct TilePlan {
    TileSize l1;
    TileSize l0;
};

constexpr uint32_t K_TO_Q_RATIO = 2U;
constexpr std::array<uint32_t, 5> TILE_CANDIDATES = {32U, 64U, 128U, 256U, 512U};
constexpr uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

inline TileSize GetL0Tile(uint32_t m, uint32_t n, uint32_t k, uint32_t dimV, const CoreMemSize& mem,
                          const TileSize& physicalL0)
{
    constexpr uint64_t ELEMENT_BYTES = 2U;
    constexpr uint64_t ACC_BYTES = 4U;
    const uint64_t l0aElems = mem.l0aBytes / 2U / ELEMENT_BYTES;
    const uint64_t l0bElems = mem.l0bBytes / 2U / ELEMENT_BYTES;
    const uint64_t l0cElems = mem.l0cBytes / ACC_BYTES;
    const uint32_t aWidth = std::max(k, n);
    const uint32_t bWidth = std::max(k, dimV);
    uint32_t l0m = std::min<uint32_t>(std::min(m, physicalL0.m), aWidth == 0U ? 0U : l0aElems / aWidth);
    uint32_t l0n = std::min<uint32_t>(std::min(n, physicalL0.n), bWidth == 0U ? 0U : l0bElems / bWidth);
    while (l0m > 0U && static_cast<uint64_t>(l0m) * l0n > l0cElems)
        l0m /= 2U;
    return {l0m, l0n, k};
}

template <class Trait>
inline TilePlan SearchFwdTileSize(uint32_t seqlenQ, uint32_t seqlenK, uint32_t batch, uint32_t heads, uint32_t dimQk,
                                  uint32_t dimV, uint32_t coreNum, const CoreMemSize& mem, bool allowDynamic = true)
{
    const uint32_t headDim = std::max(dimQk, dimV);
    const auto physical = Trait::GetFwdPhysicalCapacity(headDim);
    uint32_t blockM = physical.l1.m;
    if (allowDynamic && coreNum != 0U &&
        static_cast<uint64_t>(seqlenK) > static_cast<uint64_t>(K_TO_Q_RATIO) * seqlenQ) {
        // Prefer the largest tile that supplies enough outer work. Walking
        // from 32 upward over-splits already saturated large-batch cases and
        // doubles their Q work for no parallelism benefit.
        for (auto it = TILE_CANDIDATES.rbegin(); it != TILE_CANDIDATES.rend(); ++it) {
            const uint32_t candidate = *it;
            if (candidate > physical.l1.m)
                continue;
            blockM = candidate;
            if (CeilDiv(seqlenQ, candidate) * batch * heads >= coreNum)
                break;
        }
    }
    uint32_t blockN = physical.l1.n;
    if (allowDynamic) {
        for (auto it = TILE_CANDIDATES.rbegin(); it != TILE_CANDIDATES.rend(); ++it) {
            const uint32_t candidate = *it;
            if (candidate < blockM)
                continue;
            if (Trait::CheckFwdBuffer(blockM, candidate, dimQk, dimV, mem, physical)) {
                blockN = candidate;
                break;
            }
        }
    }
    return {{blockM, blockN, headDim}, GetL0Tile(blockM, blockN, headDim, dimV, mem, physical.l0)};
}

template <class Trait>
inline TilePlan SearchBwdTileSize(uint32_t seqlenQ, uint32_t seqlenK, uint32_t batch, uint32_t heads, uint32_t dimQk,
                                  uint32_t dimV, uint32_t coreNum, const CoreMemSize& mem, bool allowDynamic = true)
{
    const uint32_t headDim = std::max(dimQk, dimV);
    const auto physical = Trait::GetBwdPhysicalCapacity(headDim);
    uint32_t blockN = physical.l1.n;
    if (allowDynamic && coreNum != 0U && CeilDiv(seqlenK, blockN) * batch * heads < coreNum) {
        for (auto it = TILE_CANDIDATES.rbegin(); it != TILE_CANDIDATES.rend(); ++it) {
            const uint32_t candidate = *it;
            // A score/RAB row consumes one 64-element register span, making
            // 64 the lower bound for the end-to-end tile. It is not the
            // upper-bound reason: physical.l1.n remains the conservative
            // limit because N is backward's outer/distribution axis.
            if (candidate > physical.l1.n || candidate < 64U)
                continue;
            const uint32_t tail = seqlenK % candidate;
            // Avoid creating a sub-C0 outer tile solely to occupy the last
            // few cores. Such a tile performs a full synchronization round
            // for fewer than 16 rows and is both slower and numerically less
            // stable in the BF16 RAB path. The next larger candidate still
            // participates in runtime K-axis search.
            if (candidate < physical.l1.n && tail != 0U && tail < 16U)
                continue;
            blockN = candidate;
            if (CeilDiv(seqlenK, candidate) * batch * heads >= coreNum)
                break;
        }
    }
    uint32_t blockM = physical.l1.m;
    const uint64_t outerTasks = CeilDiv(seqlenK, blockN) * batch * heads;
    if (allowDynamic && coreNum != 0U && outerTasks < coreNum) {
        // A candidate larger than the whole Q sequence cannot reduce the
        // inner-loop count. Keep at least the physical tile, but avoid a
        // wider logical/packed extent when it brings no scheduling benefit.
        // Measured 74-case data also shows that widening M after the outer K
        // schedule already saturates all cores only increases inner-loop
        // latency (12-25% for M=512), so search above the physical M only in
        // the under-filled regime where fewer inner iterations are useful.
        const uint32_t usefulLimit = std::max(seqlenQ, physical.l1.m);
        for (auto it = TILE_CANDIDATES.rbegin(); it != TILE_CANDIDATES.rend(); ++it) {
            const uint32_t candidate = *it;
            if (candidate > usefulLimit || candidate < blockN)
                continue;
            // Do not trade away active cores merely to shorten the inner
            // loop.  Pick the widest candidate that still fills the device;
            // if even the physical tile cannot do so, retain the physical
            // shape as the measured low-workload optimum.
            if (candidate > physical.l1.m && outerTasks * CeilDiv(seqlenQ, candidate) < coreNum)
                continue;
            if (Trait::CheckBwdBuffer(candidate, blockN, dimQk, dimV, mem, physical)) {
                blockM = candidate;
                break;
            }
        }
    }
    return {{blockM, blockN, headDim}, GetL0Tile(blockM, blockN, headDim, dimV, mem, physical.l0)};
}

}  // namespace ops_rec::attn_plan
