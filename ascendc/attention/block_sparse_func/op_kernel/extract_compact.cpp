/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <simt_api/common_functions.h>
#include "kernel_operator.h"

using namespace AscendC;
namespace {
constexpr uint32_t BLOCK_THREAD_NUM_256 = 256;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void extract_compact_kernel_vf(
    const __gm__ int32_t* block_id, const __gm__ int32_t* mask_cnt, const __gm__ int32_t* full_cnt,
    const __gm__ int32_t* mask_offset, const __gm__ int32_t* full_offset, int32_t n_blocks, int32_t max_blocks,
    __gm__ int32_t* mask_compact, __gm__ int32_t* full_compact)
{
    // if n_blocks > 65535, need use for loop, every loop process gridDim.x blocks
    for (int32_t row = AscendC::Simt::GetBlockIdx(); row < n_blocks; row += AscendC::Simt::GetBlockNum()) {
        int32_t tid = AscendC::Simt::GetThreadIdx();
        int32_t fcnt = full_cnt[row];
        int32_t mcnt = mask_cnt[row];
        int32_t foff = full_offset[row];
        int32_t moff = mask_offset[row];

        const __gm__ int32_t* row_ptr = block_id + row * max_blocks;

        for (int32_t i = tid; i < fcnt; i += AscendC::Simt::GetThreadNum()) {
            full_compact[foff + i] = row_ptr[i];
        }
        for (int32_t i = tid; i < mcnt; i += AscendC::Simt::GetThreadNum()) {
            mask_compact[moff + i] = row_ptr[max_blocks - 1 - i];
        }
    }
}

extern "C" __global__ __aicore__ void extract_compact(GM_ADDR block_id, GM_ADDR mask_cnt, GM_ADDR full_cnt,
                                                      GM_ADDR mask_offset, GM_ADDR full_offset, GM_ADDR mask_compact,
                                                      GM_ADDR full_compact, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int32_t nBlocks = tilingData.nBlocks;
    int32_t maxBlocks = tilingData.maxBlocks;

    asc_vf_call<extract_compact_kernel_vf>(dim3{BLOCK_THREAD_NUM_256}, (__gm__ int32_t*)block_id,
                                           (__gm__ int32_t*)mask_cnt, (__gm__ int32_t*)full_cnt,
                                           (__gm__ int32_t*)mask_offset, (__gm__ int32_t*)full_offset, nBlocks,
                                           maxBlocks, (__gm__ int32_t*)mask_compact, (__gm__ int32_t*)full_compact);
}