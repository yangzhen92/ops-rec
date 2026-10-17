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

#ifndef ASCENDC_BUILD_BLOCK_DIAGONAL_MASK_ARBITRARY_FUNC_KERNEL_H_
#define ASCENDC_BUILD_BLOCK_DIAGONAL_MASK_ARBITRARY_FUNC_KERNEL_H_

#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildBlockDiagonalMaskArbitraryFunc {

constexpr uint32_t BLOCK_THREAD_NUM_1024 = 1024;
constexpr uint32_t N_FUNC = 2;

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_1024) inline void SimtBuildBlockDiagonalMaskArbitraryFunc(
    __gm__ int32_t* __restrict__ output, const __gm__ int32_t* __restrict__ offsets, int32_t totalSeqLen,
    int32_t batchSize, const uint32_t totalThread)
{
    int32_t globalPos = static_cast<int32_t>(blockId * blockDim.x + threadIdx.x);
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
        __gm__ int32_t* row = output + globalPos * N_FUNC;
        row[0] = seqStart;
        row[1] = seqEnd;
    }
}

}  // namespace BuildBlockDiagonalMaskArbitraryFunc

#endif  // ASCENDC_BUILD_BLOCK_DIAGONAL_MASK_ARBITRARY_FUNC_KERNEL_H_
