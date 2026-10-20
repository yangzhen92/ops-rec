#ifndef OPS_REC_PACKED_BLOCK_SPARSE_COMMON_H
#define OPS_REC_PACKED_BLOCK_SPARSE_COMMON_H

#include <cstdint>
#include <climits>
#include <simt_api/common_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/device_functions.h>

namespace packed_block_sparse {
constexpr int32_t THREADS = 256;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t WARPS = THREADS / WARP_SIZE;
constexpr uint32_t DESC_EMPTY = 0U;
constexpr uint32_t DESC_FULL = 1U;
constexpr uint32_t DESC_PARTIAL = 3U;
__forceinline__ __simt_callee__ uint32_t WarpOr(uint32_t value)
{
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value |= asc_shfl_down(value, offset);
    }
    return asc_shfl(value, 0);
}

// Token state bit mask: bit0=empty, bit1=partial, bit2=full.
__forceinline__ __simt_callee__ uint32_t WarpBlockState(uint32_t localMask)
{
    const uint32_t merged = WarpOr(localMask);
    if (merged == 1U) {
        return DESC_EMPTY;
    }
    if (merged == 4U) {
        return DESC_FULL;
    }
    return DESC_PARTIAL;
}

__forceinline__ __simt_callee__ uint32_t RangeBits(int32_t begin, int32_t end, int32_t wordBase)
{
    const int32_t left = begin > wordBase ? begin - wordBase : 0;
    const int32_t right = end < wordBase + 32 ? end - wordBase : 32;
    if (left >= right) {
        return 0U;
    }
    const uint32_t high = right == 32 ? 0xFFFFFFFFU : ((1U << right) - 1U);
    const uint32_t low = left == 0 ? 0U : ((1U << left) - 1U);
    return high & ~low;
}

template <int32_t N_FUNC>
__forceinline__ __simt_callee__ void LoadFunc(const __gm__ int32_t* src, int32_t strideF, int32_t* regs)
{
    if constexpr (N_FUNC > 0) {
#pragma unroll
        for (int32_t f = 0; f < N_FUNC; ++f) {
            regs[f] = src[f * strideF];
        }
    }
}

template <int32_t N_FUNC>
__forceinline__ __simt_callee__ uint32_t TokenState(const int32_t* regs, const __gm__ int32_t* src, int32_t nFunc,
                                                    int32_t strideF, int32_t kvLeft, int32_t kvRight)
{
    if constexpr (N_FUNC > 0) {
        constexpr int32_t intervals = N_FUNC / 2;
        int32_t candidate = 0;
#pragma unroll
        for (int32_t i = 0; i < intervals - 1; ++i) {
            candidate += kvLeft >= regs[2 * i + 1];
        }
        int32_t begin = regs[0];
        int32_t end = regs[1];
#pragma unroll
        for (int32_t i = 1; i < intervals; ++i) {
            begin = candidate >= i ? regs[2 * i] : begin;
            end = candidate >= i ? regs[2 * i + 1] : end;
        }
        if (begin <= kvLeft && end >= kvRight)
            return 2U;
        return begin < kvRight && end > kvLeft ? 1U : 0U;
    }
    bool any = false;
    bool full = false;
    const int32_t count = nFunc;
#pragma unroll
    for (int32_t f = 0; f + 1 < count; f += 2) {
        const int32_t begin = src[f * strideF];
        const int32_t end = src[(f + 1) * strideF];
        if (end <= begin) {
            continue;
        }
        full |= begin <= kvLeft && end >= kvRight;
        any |= begin < kvRight && end > kvLeft;
    }
    return full ? 2U : (any ? 1U : 0U);
}

template <int32_t N_FUNC>
__forceinline__ __simt_callee__ void TokenBitmap(const int32_t* regs, const __gm__ int32_t* src, int32_t nFunc,
                                                 int32_t strideF, int32_t kvLeft, int32_t kvRight, uint32_t (&bits)[4])
{
    bits[0] = bits[1] = bits[2] = bits[3] = 0U;
    const int32_t count = N_FUNC > 0 ? N_FUNC : nFunc;
#pragma unroll
    for (int32_t f = 0; f + 1 < count; f += 2) {
        const int32_t begin = N_FUNC > 0 ? regs[f] : src[f * strideF];
        const int32_t end = N_FUNC > 0 ? regs[f + 1] : src[(f + 1) * strideF];
        const int32_t clippedBegin = max(begin, kvLeft) - kvLeft;
        const int32_t clippedEnd = min(end, kvRight) - kvLeft;
        if (clippedBegin < clippedEnd) {
#pragma unroll
            for (int32_t word = 0; word < 4; ++word) {
                bits[word] |= RangeBits(clippedBegin, clippedEnd, word * 32);
            }
        }
    }
}

}  // namespace packed_block_sparse
#endif
