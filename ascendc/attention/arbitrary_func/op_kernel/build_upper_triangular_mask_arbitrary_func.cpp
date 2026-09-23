/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildUpperTriangularMaskArbitraryFunc {
constexpr uint32_t THREADS = 1024;
constexpr int32_t N_FUNC = 2;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Build(
    __gm__ int32_t* __restrict__ output,
    const __gm__ int32_t* __restrict__ offsets,
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
        begin0 = globalPos;
        end0 = seqEnd;
        __gm__ int32_t* row = output + globalPos * N_FUNC;
        row[0] = begin0;
        row[1] = end0;
    }
}
}  // namespace BuildUpperTriangularMaskArbitraryFunc

extern "C" __global__ __aicore__ void build_upper_triangular_mask_arbitrary_func(
    GM_ADDR offsets, GM_ADDR arbitraryFunc, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const uint32_t totalThreads = BuildUpperTriangularMaskArbitraryFunc::THREADS * AscendC::GetBlockNum();
    AscendC::Simt::VF_CALL<BuildUpperTriangularMaskArbitraryFunc::Build>(
        AscendC::Simt::Dim3{BuildUpperTriangularMaskArbitraryFunc::THREADS, 1, 1},
        reinterpret_cast<__gm__ int32_t*>(arbitraryFunc),
        reinterpret_cast<const __gm__ int32_t*>(offsets),
        tilingData.totalSeqLen, tilingData.batchSize, AscendC::GetBlockIdx(), totalThreads);
}
