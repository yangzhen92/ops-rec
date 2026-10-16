/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

#include <cstdint>
#include "catlass/detail/macros.hpp"

namespace Catlass::Gemm::Block::detail {

template <class T, uint32_t Elements, uint32_t Offset = 0, uint32_t Count = 1>
struct BufferPosition {
    static_assert(Count >= 1U && Count <= 3U, "buffer count must be in [1, 3]");
    static constexpr uint32_t bytes = Elements * sizeof(T);
    static constexpr uint32_t upperBound = Offset;
    static constexpr uint32_t lowerBound = Offset + bytes * Count;

    template <uint32_t Index>
    static constexpr uint32_t Get()
    {
        static_assert(Index < Count, "buffer index out of range");
        return Offset + bytes * Index;
    }
};

// Shared forward L1 layout. Runtime blockN may be wider than the compile-time
// policy, so all Cube/Vector participants must derive identical byte offsets
// from the tiling data instead of their individual static TileBuffer aliases.
template <class Element>
struct ForwardRuntimeL1Layout {
    uint32_t query[2];
    uint32_t key[2];
    uint32_t value[2];
    uint32_t prob[3];

    CATLASS_DEVICE
    ForwardRuntimeL1Layout(uint32_t blockM, uint32_t blockN, uint32_t dimQk, uint32_t dimV)
    {
        const uint32_t blockK = dimQk > dimV ? dimQk : dimV;
        const uint32_t queryBytes = blockM * blockK * sizeof(Element);
        const uint32_t keyBytes = blockN * blockK * sizeof(Element);
        const uint32_t valueBytes = blockN * blockK * sizeof(Element);
        const uint32_t probBytes = blockM * blockN * sizeof(Element);
        query[0] = 0U;
        query[1] = queryBytes;
        key[0] = 2U * queryBytes;
        key[1] = key[0] + keyBytes;
        value[0] = key[0] + 2U * keyBytes;
        value[1] = value[0] + valueBytes;
        prob[0] = value[0] + 2U * valueBytes;
        prob[1] = prob[0] + probBytes;
        prob[2] = prob[1] + probBytes;
    }
};

template <class Element, class Accumulator>
struct ForwardRuntimeBufferLayout {
    static constexpr uint32_t STAGES = 3U;
    static constexpr uint32_t AIV_PER_AIC = 2U;
    ForwardRuntimeL1Layout<Element> l1;
    uint32_t l0aQ{0U};
    uint32_t l0aP{0U};
    uint32_t l0bK{0U};
    uint32_t l0bV{0U};
    uint32_t l0cQk{0U};
    uint32_t l0cPv{0U};
    uint32_t score[3];
    uint32_t prob[STAGES];
    uint32_t rab[2];
    uint32_t transOut{0U};
    uint32_t transIn[2];
    uint32_t packedMask{0U};

    CATLASS_DEVICE
    ForwardRuntimeBufferLayout(uint32_t blockM, uint32_t blockN, uint32_t dimQk, uint32_t dimV)
        : l1(blockM, blockN, dimQk, dimV)
    {
        const uint32_t blockK = dimQk > dimV ? dimQk : dimV;
        const uint32_t qkL0aBytes = blockM * blockK * sizeof(Element);
        const uint32_t qkL0bBytes = blockN * blockK * sizeof(Element);
        const uint32_t qkL0cBytes = blockM * blockN * sizeof(Accumulator);
        l0aP = qkL0aBytes;
        l0bV = qkL0bBytes;
        l0cPv = qkL0cBytes;

        const uint32_t splitMnElements = blockM * blockN / AIV_PER_AIC;
        const uint32_t splitMkElements = blockM * blockK / AIV_PER_AIC;
        const uint32_t scoreElements = splitMnElements > splitMkElements ? splitMnElements : splitMkElements;
        const uint32_t scoreBytes = scoreElements * sizeof(Element);
        const uint32_t probBytes = splitMnElements * sizeof(Accumulator);
        const uint32_t rabBytes = splitMnElements * sizeof(Element);
        const uint32_t transOutBytes = splitMkElements * sizeof(Element);
        const uint32_t transInBytes = splitMkElements * sizeof(Accumulator);
        score[0] = 0U;
        score[1] = scoreBytes;
        score[2] = 0U;
        prob[0] = 2U * scoreBytes;
        prob[1] = prob[0] + probBytes;
        prob[2] = prob[1] + probBytes;
        rab[0] = prob[0] + STAGES * probBytes;
        rab[1] = rab[0] + rabBytes;
        transOut = rab[0] + 2U * rabBytes;
        transIn[0] = transOut + transOutBytes;
        transIn[1] = transIn[0] + transInBytes;
        packedMask = transIn[0] + 2U * transInBytes;
    }
};

}  // namespace Catlass::Gemm::Block::detail
