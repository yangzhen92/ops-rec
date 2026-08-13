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

__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void dual_exclusive_scan_kernel_vf(__gm__ int32_t* mask_cnt,
                                                                                 __gm__ int32_t* full_cnt,
                                                                                 __gm__ int32_t* mask_offset,
                                                                                 __gm__ int32_t* full_offset, int32_t n)
{
    // n is tiny for inference (B*H*num_blocks ≤ 32); single-thread serial scan
    // has the lowest launch+compute overhead.
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int32_t m_acc = 0, f_acc = 0;
    mask_offset[0] = 0;
    full_offset[0] = 0;
#pragma unroll 32
    for (int32_t i = 0; i < n; ++i) {
        m_acc += mask_cnt[i];
        f_acc += full_cnt[i];
        mask_offset[i + 1] = m_acc;
        full_offset[i + 1] = f_acc;
    }
}

extern "C" __global__ __aicore__ void dual_exclusive_scan(GM_ADDR mask_cnt, GM_ADDR full_cnt, GM_ADDR mask_offset,
                                                          GM_ADDR full_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int32_t totalLength = tilingData.totalLength;

    asc_vf_call<dual_exclusive_scan_kernel_vf>(dim3{1}, (__gm__ int32_t*)mask_cnt, (__gm__ int32_t*)full_cnt,
                                               (__gm__ int32_t*)mask_offset, (__gm__ int32_t*)full_offset, totalLength);
}
