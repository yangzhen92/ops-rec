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

#ifndef ASCENDC_BUILD_GIST_MASK_ARBITRARY_FUNC_KERNEL_H_
#define ASCENDC_BUILD_GIST_MASK_ARBITRARY_FUNC_KERNEL_H_

#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildGistMaskArbitraryFunc {

constexpr uint32_t BLOCK_THREAD_NUM_1024 = 1024;
constexpr int32_t N_FUNC = 4;

__forceinline__ __simt_callee__ void AppendInterval(int32_t begin, int32_t end, int32_t seqStart, int32_t seqEnd,
                                                    int32_t* intervals, int32_t& count)
{
    begin = max(begin, seqStart);
    end = min(end, seqEnd);
    if (begin >= end) {
        return;
    }
    if (count >= N_FUNC / 2) {
        if (count > 0 && intervals[(count - 1) * 2 + 1] == begin) {
            intervals[(count - 1) * 2 + 1] = end;
        }
        return;
    }
    if (count > 0 && intervals[(count - 1) * 2 + 1] == begin) {
        intervals[(count - 1) * 2 + 1] = end;
        return;
    }
    intervals[count * 2] = begin;
    intervals[count * 2 + 1] = end;
    ++count;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_1024) inline void SimtBuildGistMaskArbitraryFunc(
    __gm__ int32_t* __restrict__ output, const __gm__ int32_t* __restrict__ offsets,
    const __gm__ int32_t* __restrict__ preDocLens, const __gm__ int32_t* __restrict__ gistStarts,
    const __gm__ int32_t* __restrict__ gistEnds, int32_t totalSeqLen, int32_t batchSize, const uint32_t totalThread)
{
    int32_t globalPos = blockIdx.x * blockDim.x + threadIdx.x;
    for (; globalPos < totalSeqLen; globalPos += static_cast<int32_t>(totalThread)) {
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

#endif  // ASCENDC_BUILD_GIST_MASK_ARBITRARY_FUNC_KERNEL_H_
