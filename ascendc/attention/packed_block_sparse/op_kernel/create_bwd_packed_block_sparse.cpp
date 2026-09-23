/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include "kernel_operator.h"
#include "packed_block_sparse_common.h"

using namespace AscendC;
using namespace packed_block_sparse;

namespace {
template <int32_t N_FUNC, int32_t Q_BLOCK, int32_t KV_BLOCK, bool K2Q>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CreatePackedVf(
    const __gm__ int32_t* func, const __gm__ int32_t* qOffsets, const __gm__ int32_t* kvOffsets,
    __gm__ int32_t* maskCnt, __gm__ int32_t* fullCnt, __gm__ int32_t* rowBounds,
    __gm__ int32_t* descriptor, __gm__ int32_t* bitmap, int32_t strideH,
    int32_t strideQ, int32_t strideF, int32_t nFunc,
    int32_t numQBlocks, int32_t numKvBlocks, int32_t hNum, int32_t bNum)
{
    constexpr int32_t BITMAP_WORDS = KV_BLOCK / 32;
    constexpr int32_t TOKENS_PER_LANE = Q_BLOCK / WARP_SIZE;
    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t lane = tid & (WARP_SIZE - 1);
    const int32_t warp = tid / WARP_SIZE;
    const int32_t outerBlocks = K2Q ? numKvBlocks : numQBlocks;
    const int32_t innerBlocks = K2Q ? numQBlocks : numKvBlocks;
    const int32_t totalOuter = bNum * hNum * outerBlocks;
    __ubuf__ int32_t partialCounts[WARPS];
    __ubuf__ int32_t fullCounts[WARPS];
    __ubuf__ int32_t firstBlocks[WARPS];
    __ubuf__ int32_t lastBlocks[WARPS];

    for (int32_t outer = AscendC::Simt::GetBlockIdx(); outer < totalOuter;
         outer += AscendC::Simt::GetBlockNum()) {
        const int32_t bh = outer / outerBlocks;
        const int32_t outerBlock = outer - bh * outerBlocks;
        const int32_t b = bh / hNum;
        const int32_t h = bh - b * hNum;
        const int32_t qLen = qOffsets[b + 1] - qOffsets[b];
        const int32_t kvStartGlobal = kvOffsets[b];
        const int32_t kvLen = kvOffsets[b + 1] - kvStartGlobal;
        const int32_t funcQStartGlobal = kvOffsets[b + 1] - qLen;
        const int32_t qBlock = K2Q ? 0 : outerBlock;
        const int32_t qStartBase = qBlock * Q_BLOCK;
        const __gm__ int32_t* funcHead = func + h * strideH;
        int32_t partialCount = 0;
        int32_t fullCount = 0;
        int32_t firstBlock = innerBlocks;
        int32_t lastBlock = -1;
        int32_t regs[TOKENS_PER_LANE][N_FUNC > 0 ? N_FUNC : 1];
        if constexpr (!K2Q && N_FUNC > 0) {
#pragma unroll
            for (int32_t token = 0; token < TOKENS_PER_LANE; ++token) {
                const int32_t q = qStartBase + lane + token * WARP_SIZE;
                if (q < qLen) {
                    LoadFunc<N_FUNC>(funcHead + (funcQStartGlobal + q) * strideQ, strideF, regs[token]);
                }
            }
        }

        for (int32_t inner = warp; inner < innerBlocks; inner += WARPS) {
            const int32_t currentQBlock = K2Q ? inner : outerBlock;
            const int32_t currentKvBlock = K2Q ? outerBlock : inner;
            const int32_t qStart = K2Q ? currentQBlock * Q_BLOCK : qStartBase;
            const int32_t kvLeft = currentKvBlock * KV_BLOCK;
            const int32_t kvRight = min(kvLeft + KV_BLOCK, kvLen);
            const int32_t kvLeftGlobal = kvStartGlobal + kvLeft;
            const int32_t kvRightGlobal = kvStartGlobal + kvRight;
            uint32_t localStateMask = 0U;

#pragma unroll
            for (int32_t token = 0; token < TOKENS_PER_LANE; ++token) {
                const int32_t q = qStart + lane + token * WARP_SIZE;
                if (q < qLen && kvLeft < kvLen) {
                    const __gm__ int32_t* row = funcHead + (funcQStartGlobal + q) * strideQ;
                    if constexpr (K2Q && N_FUNC > 0) {
                        LoadFunc<N_FUNC>(row, strideF, regs[token]);
                    }
                    const uint32_t tokenState = TokenState<N_FUNC>(
                        regs[token], row, nFunc, strideF, kvLeftGlobal, kvRightGlobal);
                    localStateMask |= 1U << tokenState;
                } else {
                    localStateMask |= 1U;
                }
            }

            // A short KV tail cannot be classified FULL because HSTU writes a
            // complete 128-column score tile. Force it through the bitmap path.
            if (kvLeft < kvLen && kvRight - kvLeft != KV_BLOCK) {
                localStateMask |= 1U;
            }
            const uint32_t state = WarpBlockState(localStateMask);
            const int32_t descriptorPos = outer * innerBlocks + inner;
            if (lane == 0) {
                descriptor[descriptorPos] = static_cast<int32_t>((static_cast<uint32_t>(inner) << 2) | state);
                fullCount += state == DESC_FULL;
                if (state != DESC_EMPTY) {
                    firstBlock = min(firstBlock, inner);
                    lastBlock = max(lastBlock, inner);
                }
            }

            if (state == DESC_PARTIAL) {
                const int64_t slot = static_cast<int64_t>(outer) * innerBlocks + inner;
#pragma unroll
                for (int32_t token = 0; token < TOKENS_PER_LANE; ++token) {
                    const int32_t rowInBlock = lane + token * WARP_SIZE;
                    const int32_t q = qStart + rowInBlock;
                    uint32_t bits[4] = {0U, 0U, 0U, 0U};
                    if (q < qLen && kvLeft < kvLen) {
                        const __gm__ int32_t* row = funcHead + (funcQStartGlobal + q) * strideQ;
                        TokenBitmap<N_FUNC>(
                            regs[token], row, nFunc, strideF, kvLeftGlobal, kvRightGlobal, bits);
                    }
                    const int64_t bitmapPos = (slot * Q_BLOCK + rowInBlock) * BITMAP_WORDS;
#pragma unroll
                    for (int32_t word = 0; word < BITMAP_WORDS; ++word) {
                        bitmap[bitmapPos + word] = static_cast<int32_t>(bits[word]);
                    }
                }
                ++partialCount;
            }
        }
        if (lane == 0) {
            partialCounts[warp] = partialCount;
            fullCounts[warp] = fullCount;
            firstBlocks[warp] = firstBlock;
            lastBlocks[warp] = lastBlock;
        }
        asc_syncthreads();
        if (tid == 0) {
            int32_t partialTotal = 0;
            int32_t fullTotal = 0;
            int32_t first = innerBlocks;
            int32_t last = -1;
#pragma unroll
            for (int32_t i = 0; i < WARPS; ++i) {
                partialTotal += partialCounts[i];
                fullTotal += fullCounts[i];
                first = min(first, firstBlocks[i]);
                last = max(last, lastBlocks[i]);
            }
            maskCnt[outer] = partialTotal;
            fullCnt[outer] = fullTotal;
            rowBounds[outer * 2] = first == innerBlocks ? -1 : first;
            rowBounds[outer * 2 + 1] = last;
        }
        asc_syncthreads();
    }
}

#define LAUNCH_PACKED(NF, QB, KB, K2Q_MODE) \
    asc_vf_call<CreatePackedVf<NF, QB, KB, K2Q_MODE>>(dim3{256}, f, qOffsets, kvOffsets, mc, fc, bounds, desc, bm, t.strideH, \
        t.strideQ, t.strideF, t.nFunc, t.numQBlocks, t.numKvBlocks, t.hNum, t.bNum)

#define DISPATCH_PACKED(QB, KB, K2Q_MODE) \
    switch (t.nFunc) { \
        case 2: LAUNCH_PACKED(2, QB, KB, K2Q_MODE); break; \
        case 4: LAUNCH_PACKED(4, QB, KB, K2Q_MODE); break; \
        case 6: LAUNCH_PACKED(6, QB, KB, K2Q_MODE); break; \
        case 8: LAUNCH_PACKED(8, QB, KB, K2Q_MODE); break; \
        default: LAUNCH_PACKED(0, QB, KB, K2Q_MODE); break; \
    }
}  // namespace

extern "C" __global__ __aicore__ void create_bwd_packed_block_sparse(
    GM_ADDR func, GM_ADDR qOffsetsAddr, GM_ADDR kvOffsetsAddr, GM_ADDR maskCnt, GM_ADDR fullCnt, GM_ADDR rowBounds,
    GM_ADDR descriptor, GM_ADDR bitmap, GM_ADDR, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t, tiling);
    auto* f = reinterpret_cast<__gm__ int32_t*>(func);
    auto* qOffsets = reinterpret_cast<__gm__ int32_t*>(qOffsetsAddr);
    auto* kvOffsets = reinterpret_cast<__gm__ int32_t*>(kvOffsetsAddr);
    auto* mc = reinterpret_cast<__gm__ int32_t*>(maskCnt);
    auto* fc = reinterpret_cast<__gm__ int32_t*>(fullCnt);
    auto* bounds = reinterpret_cast<__gm__ int32_t*>(rowBounds);
    auto* desc = reinterpret_cast<__gm__ int32_t*>(descriptor);
    auto* bm = reinterpret_cast<__gm__ int32_t*>(bitmap);
    if (t.qBlockSize == 128 && t.kvBlockSize == 64) {
        DISPATCH_PACKED(128, 64, true);
    } else {
        DISPATCH_PACKED(256, 128, true);
    }
}
