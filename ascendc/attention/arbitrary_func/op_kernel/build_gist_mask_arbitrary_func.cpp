/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildGistMaskArbitraryFunc {
constexpr uint32_t THREADS = 1024;
constexpr int32_t N_FUNC = 4;

__forceinline__ __simt_callee__ void AppendInterval(
    int32_t begin, int32_t end, int32_t seqStart, int32_t seqEnd, int32_t* intervals, int32_t& count)
{
    begin = max(begin, seqStart);
    end = min(end, seqEnd);
    if (begin >= end) {
        return;
    }
    if (count > 0 && intervals[(count - 1) * 2 + 1] == begin) {
        intervals[(count - 1) * 2 + 1] = end;
        return;
    }
    if (count >= N_FUNC / 2) {
        return;
    }
    intervals[count * 2] = begin;
    intervals[count * 2 + 1] = end;
    ++count;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Build(
    __gm__ int32_t* __restrict__ output,
    const __gm__ int32_t* __restrict__ offsets,
    const __gm__ int32_t* __restrict__ preDocLens,
    const __gm__ int32_t* __restrict__ gistStarts,
    const __gm__ int32_t* __restrict__ gistEnds,
    int32_t totalSeqLen, int32_t batchSize, uint32_t blockId, uint32_t totalThreads)
{
    int32_t globalPos = static_cast<int32_t>(blockId * blockDim.x + threadIdx.x);
    for (; globalPos < totalSeqLen; globalPos += static_cast<int32_t>(totalThreads)) {
        int32_t low = 0;
        int32_t high = batchSize;
        while (low < high) {
            const int32_t middle = (low + high) >> 1;
            if (offsets[middle + 1] <= globalPos) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        const int32_t seqId = low;
        const int32_t seqStart = offsets[seqId];
        const int32_t seqEnd = offsets[seqId + 1];
        const int32_t seqLen = seqEnd - seqStart;
        const int32_t pos = globalPos - seqStart;
        const int32_t p = min(max(preDocLens[seqId], 0), seqLen);
        const int32_t g0 = min(max(gistStarts[seqId], p), seqLen);
        const int32_t g1 = min(max(gistEnds[seqId], g0), seqLen);
        const int32_t pGlobal = seqStart + p;
        const int32_t g0Global = seqStart + g0;
        const int32_t g1Global = seqStart + g1;

        int32_t intervals[N_FUNC] = {seqStart, seqStart, seqStart, seqStart};
        int32_t count = 0;
        if (pos < p) {
            AppendInterval(seqStart, globalPos + 1, seqStart, seqEnd, intervals, count);
        } else if (pos < g0) {
            AppendInterval(seqStart, pGlobal, seqStart, seqEnd, intervals, count);
            AppendInterval(pGlobal, globalPos + 1, seqStart, seqEnd, intervals, count);
        } else if (pos < g1) {
            AppendInterval(seqStart, globalPos + 1, seqStart, seqEnd, intervals, count);
        } else {
            AppendInterval(seqStart, pGlobal, seqStart, seqEnd, intervals, count);
            AppendInterval(g0Global, g1Global, seqStart, seqEnd, intervals, count);
            AppendInterval(g1Global, globalPos + 1, seqStart, seqEnd, intervals, count);
        }
        for (int32_t i = count; i < N_FUNC / 2; ++i) {
            const int32_t last = i == 0 ? seqStart : intervals[i * 2 - 1];
            intervals[i * 2] = last;
            intervals[i * 2 + 1] = last;
        }
        __gm__ int32_t* row = output + globalPos * N_FUNC;
        *reinterpret_cast<__gm__ int4*>(row) = int4{intervals[0], intervals[1], intervals[2], intervals[3]};
    }
}
}  // namespace BuildGistMaskArbitraryFunc

extern "C" __global__ __aicore__ void build_gist_mask_arbitrary_func(
    GM_ADDR offsets, GM_ADDR preDocLens, GM_ADDR gistStarts, GM_ADDR gistEnds,
    GM_ADDR arbitraryFunc, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const uint32_t totalThreads = BuildGistMaskArbitraryFunc::THREADS * AscendC::GetBlockNum();
    AscendC::Simt::VF_CALL<BuildGistMaskArbitraryFunc::Build>(
        AscendC::Simt::Dim3{BuildGistMaskArbitraryFunc::THREADS, 1, 1},
        reinterpret_cast<__gm__ int32_t*>(arbitraryFunc),
        reinterpret_cast<const __gm__ int32_t*>(offsets),
        reinterpret_cast<const __gm__ int32_t*>(preDocLens),
        reinterpret_cast<const __gm__ int32_t*>(gistStarts),
        reinterpret_cast<const __gm__ int32_t*>(gistEnds),
        tilingData.totalSeqLen, tilingData.batchSize, AscendC::GetBlockIdx(), totalThreads);
}
