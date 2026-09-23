/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildTargetGroupBlockMaskArbitraryFunc {
constexpr uint32_t THREADS = 1024;
constexpr int32_t N_FUNC = 4;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Build(
    __gm__ int32_t* __restrict__ output,
    const __gm__ int32_t* __restrict__ offsets,
    const __gm__ int32_t* __restrict__ targetLens,
    const __gm__ int32_t* __restrict__ groupSizes,
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
        int32_t begin0 = seqStart;
        int32_t end0 = seqStart;
        int32_t begin1 = seqStart;
        int32_t end1 = seqStart;

        const int32_t target = min(max(targetLens[seqId], 0), seqLen);
        const int32_t history = seqLen - target;
        const int32_t group = max(groupSizes[seqId], 1);
        if (pos < history) {
            end0 = globalPos + 1;
        } else {
            end0 = seqStart + history;
            const int32_t targetPos = pos - history;
            const int32_t groupStart = (targetPos / group) * group;
            begin1 = seqStart + history + groupStart;
            end1 = seqStart + history + min(groupStart + group, target);
        }

        if (begin1 == end1) {
            begin1 = end0;
            end1 = end0;
        }
        __gm__ int32_t* row = output + globalPos * N_FUNC;
        *reinterpret_cast<__gm__ int4*>(row) = int4{begin0, end0, begin1, end1};
    }
}
}  // namespace BuildTargetGroupBlockMaskArbitraryFunc

extern "C" __global__ __aicore__ void build_target_group_block_mask_arbitrary_func(
    GM_ADDR offsets, GM_ADDR targetLens, GM_ADDR groupSizes, GM_ADDR arbitraryFunc, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const uint32_t totalThreads = BuildTargetGroupBlockMaskArbitraryFunc::THREADS * AscendC::GetBlockNum();
    AscendC::Simt::VF_CALL<BuildTargetGroupBlockMaskArbitraryFunc::Build>(
        AscendC::Simt::Dim3{BuildTargetGroupBlockMaskArbitraryFunc::THREADS, 1, 1},
        reinterpret_cast<__gm__ int32_t*>(arbitraryFunc),
        reinterpret_cast<const __gm__ int32_t*>(offsets),
        reinterpret_cast<const __gm__ int32_t*>(targetLens),
        reinterpret_cast<const __gm__ int32_t*>(groupSizes),
        tilingData.totalSeqLen, tilingData.batchSize, AscendC::GetBlockIdx(), totalThreads);
}
