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
constexpr uint32_t BLOCK_THREAD_NUM = 1024;

// 展开输出区间 [begin, end)：offsets 为各行前缀和（长 n_blocks+1，单调非降）。
// reverse=1 时行内取尾部倒数第 idx 个元素（mask 路），否则取头部第 idx 个（full 路）。
__forceinline__ __simt_callee__ void copy_range(const __gm__ int32_t* block_id, const __gm__ int32_t* offsets,
                                                int32_t n_blocks, int32_t max_blocks, __gm__ int32_t* dst,
                                                int32_t begin, int32_t end, int32_t reverse)
{
    if (begin >= end) {
        return;
    }

    // 二分定位起始行：最大 row 使 offsets[row] <= begin（offsets[0]=0，必然存在）
    int32_t lo = 0;
    int32_t hi = n_blocks;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo + 1) / 2;
        if (offsets[mid] <= begin) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int32_t row = lo;

    // 行号随 pos 顺序推进，跨行连续写出
    int32_t pos = begin;
    while (pos < end) {
        const int32_t rowEnd = offsets[row + 1];
        if (pos >= rowEnd) {  // 跳过空行
            ++row;
            continue;
        }
        const int32_t take = (end < rowEnd ? end : rowEnd) - pos;
        const int32_t idx = pos - offsets[row];
        if (reverse) {
            const __gm__ int32_t* src = block_id + row * max_blocks + (max_blocks - 1 - idx);
            for (int32_t k = 0; k < take; ++k) {
                dst[pos + k] = src[-k];
            }
        } else {
            const __gm__ int32_t* src = block_id + row * max_blocks + idx;
            for (int32_t k = 0; k < take; ++k) {
                dst[pos + k] = src[k];
            }
        }
        pos += take;
        ++row;
    }
}
}  // namespace

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM) inline void extract_compact_kernel_vf(
    const __gm__ int32_t* block_id, const __gm__ int32_t* mask_offset, const __gm__ int32_t* full_offset,
    int32_t n_blocks, int32_t max_blocks, int32_t total_mask, int32_t total_full, __gm__ int32_t* mask_compact,
    __gm__ int32_t* full_compact)
{
    // 全局工作槽 = blockIdx * threadNum + threadIdx，输出区间按元素数均分给所有核所有线程：
    // 线程利用率不受行长限制，各行 cnt 方差也不再影响核间负载均衡。
    const int32_t gtid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    const int32_t gstride = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();

    {
        const int32_t chunk = (total_full + gstride - 1) / gstride;
        const int32_t begin = gtid * chunk;
        const int32_t end = begin + chunk < total_full ? begin + chunk : total_full;
        copy_range(block_id, full_offset, n_blocks, max_blocks, full_compact, begin, end, 0);
    }
    {
        const int32_t chunk = (total_mask + gstride - 1) / gstride;
        const int32_t begin = gtid * chunk;
        const int32_t end = begin + chunk < total_mask ? begin + chunk : total_mask;
        copy_range(block_id, mask_offset, n_blocks, max_blocks, mask_compact, begin, end, 1);
    }
}

extern "C" __global__ __aicore__ void extract_compact(GM_ADDR block_id, GM_ADDR mask_cnt, GM_ADDR full_cnt,
                                                      GM_ADDR mask_offset, GM_ADDR full_offset, GM_ADDR mask_compact,
                                                      GM_ADDR full_compact, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int32_t nBlocks = tilingData.nBlocks;
    int32_t maxBlocks = tilingData.maxBlocks;

    // offset 数组长 nBlocks+1，末项即元素总数（cnt 不再单独读取，offset 差即行内元素数）
    __gm__ int32_t* mo = (__gm__ int32_t*)mask_offset;
    __gm__ int32_t* fo = (__gm__ int32_t*)full_offset;
    int32_t totalMask = mo[nBlocks];
    int32_t totalFull = fo[nBlocks];

    asc_vf_call<extract_compact_kernel_vf>(dim3{BLOCK_THREAD_NUM}, (__gm__ int32_t*)block_id, mo, fo, nBlocks,
                                           maxBlocks, totalMask, totalFull, (__gm__ int32_t*)mask_compact,
                                           (__gm__ int32_t*)full_compact);
}
