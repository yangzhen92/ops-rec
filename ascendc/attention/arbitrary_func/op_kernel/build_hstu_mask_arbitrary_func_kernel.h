/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_BUILD_HSTU_MASK_ARBITRARY_FUNC_KERNEL_H_
#define ASCENDC_BUILD_HSTU_MASK_ARBITRARY_FUNC_KERNEL_H_

#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildHstuMaskArbitraryFunc {

constexpr uint32_t BLOCK_THREAD_NUM_256 = 1024;
constexpr uint32_t N_FUNC = 4;

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void SimtBuildHstuMaskArbitraryFunc(
    __gm__ int32_t* __restrict__ af,             // [total_seqlen, n_func], row-major
    const __gm__ int32_t* __restrict__ offsets,  // [batch_size + 1]
    const __gm__ int32_t* __restrict__ ctxLens,  // [batch_size]
    const __gm__ int32_t* __restrict__ tgtLens,  // [batch_size]
    int32_t targetGroupSize, int32_t totalSeqLen, int32_t batchSize, const uint32_t blockId,
    const uint32_t totalThread)
{
    int32_t pos = blockId * blockDim.x + threadIdx.x;
    if (pos >= totalSeqLen) 
        return;

    for (; pos < totalSeqLen; pos += totalThread) {
        // Binary search to find which sequence this position belongs to
        int32_t seqId = 0;
        {
            int32_t lo = 0, hi = batchSize;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if (offsets[mid + 1] <= pos) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            seqId = lo;
        }

        int32_t seqStart = offsets[seqId];
        int32_t seqEnd = offsets[seqId + 1];
        int32_t seqLen = seqEnd - seqStart;
        int32_t ctxLen = ctxLens[seqId];
        int32_t tgtLen = tgtLens[seqId];
        int32_t histLen = seqLen - ctxLen - tgtLen;

        int32_t posInSeq = pos - seqStart;
        int32_t histEnd = ctxLen + histLen;
        int32_t base = seqStart + histEnd;

        // V2 layout: af[pos][0..3] row-major, write all 4 cols in one int4 store
        __gm__ int32_t* afBase = af + pos * N_FUNC;

        int4 v;  // the 4 columns
        if (posInSeq < ctxLen) {
            // Context token: single interval [seq_start, base), col2/3 = 0
            v.x = seqStart;
            v.y = base;
            v.z = 0;
            v.w = 0;
        } else if (posInSeq < histEnd) {
            // History token: single interval [seq_start, pos+1), col2/3 = 0
            v.x = seqStart;
            v.y = pos + 1;
            v.z = 0;
            v.w = 0;
        } else {
            // Target token: two intervals
            // interval1: [seq_start, base)
            // interval2: [base + groupStart, base + tgtIdx + 1)
            int32_t tgtIdx = posInSeq - histEnd;
            int32_t groupStart = (tgtIdx / targetGroupSize) * targetGroupSize;
            v.x = seqStart;
            v.y = base;
            v.z = base + groupStart;
            v.w = base + tgtIdx + 1;
        }

        // Single 16B vector store (n_func is 4 for this kernel)
        *reinterpret_cast<__gm__ int4*>(afBase) = v;
    }
}

}  // namespace BuildHstuMaskArbitraryFunc

#endif  // ASCENDC_BUILD_HSTU_MASK_ARBITRARY_FUNC_KERNEL_H_
