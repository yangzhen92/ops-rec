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

#include "build_hstu_mask_arbitrary_func_kernel.h"

extern "C" __global__ __aicore__ void build_hstu_mask_arbitrary_func(GM_ADDR offsets, GM_ADDR ctxLens, GM_ADDR tgtLens,
                                                                     GM_ADDR arbitraryFunc, GM_ADDR workspace,
                                                                     GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    int32_t totalSeqLen = tilingData.totalSeqLen;
    int32_t targetGroupSize = tilingData.targetGroupSize;
    int32_t batchSize = tilingData.batchSize;

    const uint32_t totalThread = BuildHstuMaskArbitraryFunc::BLOCK_THREAD_NUM_256 * AscendC::GetBlockNum();
    AscendC::Simt::VF_CALL<BuildHstuMaskArbitraryFunc::SimtBuildHstuMaskArbitraryFunc>(
        AscendC::Simt::Dim3{BuildHstuMaskArbitraryFunc::BLOCK_THREAD_NUM_256, 1, 1},
        reinterpret_cast<__gm__ int*>(arbitraryFunc), reinterpret_cast<const __gm__ int*>(offsets),
        reinterpret_cast<const __gm__ int*>(ctxLens), reinterpret_cast<const __gm__ int*>(tgtLens), targetGroupSize,
        totalSeqLen, batchSize, AscendC::GetBlockIdx(), totalThread);
}
