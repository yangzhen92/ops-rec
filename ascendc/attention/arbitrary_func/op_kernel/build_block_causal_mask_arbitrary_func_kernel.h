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

#ifndef ASCENDC_BUILD_BLOCK_CAUSAL_MASK_ARBITRARY_FUNC_KERNEL_H_
#define ASCENDC_BUILD_BLOCK_CAUSAL_MASK_ARBITRARY_FUNC_KERNEL_H_

#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildBlockCausalMaskArbitraryFunc {

constexpr uint32_t BLOCK_THREAD_NUM_1024 = 1024;
constexpr uint32_t N_FUNC = 2;

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_1024) inline void SimtBuildBlockCausalMaskArbitraryFunc(
    __gm__ int32_t* __restrict__ af,                // [total_seqlen, n_func], row-major
    const __gm__ int32_t* __restrict__ offsets,     // [batch_size + 1]
    const __gm__ int32_t* __restrict__ blockSizes,  // [batch_size]
    int32_t totalSeqLen, int32_t batchSize, const uint32_t blockId, const uint32_t totalThread)
{
    int32_t pos = static_cast<int32_t>(blockId * blockDim.x + threadIdx.x);
    for (; pos < totalSeqLen; pos += static_cast<int32_t>(totalThread)) {
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

        const int32_t seqStart = offsets[seqId];
        const int32_t seqEnd = offsets[seqId + 1];
        const int32_t seqLen = seqEnd - seqStart;
        const int32_t posInSeq = pos - seqStart;

        const int32_t blockSize = (blockSizes[seqId] > 0) ? blockSizes[seqId] : 1;
        const int32_t endIdx = seqStart + (((posInSeq / blockSize) + 1) * blockSize);
        const int32_t endClamped = (endIdx > seqEnd) ? seqEnd : endIdx;

        __gm__ int32_t* row = af + pos * N_FUNC;
        row[0] = seqStart;
        row[1] = endClamped;
    }
}

}  // namespace BuildBlockCausalMaskArbitraryFunc

#endif  // ASCENDC_BUILD_BLOCK_CAUSAL_MASK_ARBITRARY_FUNC_KERNEL_H_
