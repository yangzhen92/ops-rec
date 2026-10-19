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
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t SCAN_THREADS = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t NWARPS = static_cast<int32_t>(SCAN_THREADS) / WARP_SIZE;  // 32

__forceinline__ __simt_callee__ void warp_inclusive_dual(int32_t& m, int32_t& f, int32_t lane)
{
#pragma unroll
    for (int32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int32_t pm = asc_shfl_up(m, offset);
        int32_t pf = asc_shfl_up(f, offset);
        if (lane >= offset) {
            m += pm;
            f += pf;
        }
    }
}
}  // namespace

// 阶段一 VF：仅计算本核两路总和（每核只写 workspace 2 个 int，不触碰输出区）
__simt_vf__ __aicore__ LAUNCH_BOUND(SCAN_THREADS) inline void dual_scan_totals_vf(const __gm__ int32_t* mask_cnt,
                                                                                  const __gm__ int32_t* full_cnt,
                                                                                  int32_t len, __gm__ int32_t* mask_tot,
                                                                                  __gm__ int32_t* full_tot,
                                                                                  int32_t core_idx)
{
    const int32_t tid = static_cast<int32_t>(AscendC::Simt::GetThreadIdx());
    const int32_t threads = static_cast<int32_t>(SCAN_THREADS);
    const int32_t lane = tid & (WARP_SIZE - 1);
    const int32_t warp = tid >> 5;

    const int32_t per = (len + threads - 1) / threads;
    const int32_t lo = tid * per;
    const int32_t hi = (lo + per < len) ? (lo + per) : len;

    int32_t sm = 0;
    int32_t sf = 0;
    for (int32_t i = lo; i < hi; ++i) {
        sm += mask_cnt[i];
        sf += full_cnt[i];
    }

    // warp 内 inclusive scan 后，lane 31 即持有该 warp 的两路总和
    warp_inclusive_dual(sm, sf, lane);

    __ubuf__ int32_t warp_sum_m[NWARPS];
    __ubuf__ int32_t warp_sum_f[NWARPS];
    if (lane == WARP_SIZE - 1) {
        warp_sum_m[warp] = sm;
        warp_sum_f[warp] = sf;
    }
    asc_syncthreads();

    if (tid == 0) {
        int32_t tm = 0;
        int32_t tf = 0;
        for (int32_t w = 0; w < NWARPS; ++w) {
            tm += warp_sum_m[w];
            tf += warp_sum_f[w];
        }
        mask_tot[core_idx] = tm;
        full_tot[core_idx] = tf;
    }
}

// 阶段二 VF：本地 exclusive scan 直接叠加核间前缀，单次写出最终结果。
// cnt 数据刚被阶段一读过，读取命中 DataCache/L2，无需显式搬入。
__simt_vf__ __aicore__ LAUNCH_BOUND(SCAN_THREADS) inline void dual_scan_final_vf(
    const __gm__ int32_t* mask_cnt, const __gm__ int32_t* full_cnt, __gm__ int32_t* mask_offset,
    __gm__ int32_t* full_offset, int32_t len, int32_t pref_m, int32_t pref_f)
{
    const int32_t tid = static_cast<int32_t>(AscendC::Simt::GetThreadIdx());
    const int32_t threads = static_cast<int32_t>(SCAN_THREADS);
    const int32_t lane = tid & (WARP_SIZE - 1);
    const int32_t warp = tid >> 5;

    const int32_t per = (len + threads - 1) / threads;
    const int32_t lo = tid * per;
    const int32_t hi = (lo + per < len) ? (lo + per) : len;

    __ubuf__ int32_t warp_tot_m[NWARPS];
    __ubuf__ int32_t warp_tot_f[NWARPS];
    __ubuf__ int32_t warp_pref_m[NWARPS];
    __ubuf__ int32_t warp_pref_f[NWARPS];

    int32_t sm = 0;
    int32_t sf = 0;
    for (int32_t i = lo; i < hi; ++i) {
        sm += mask_cnt[i];
        sf += full_cnt[i];
    }

    const int32_t m_lane0 = sm;
    const int32_t f_lane0 = sf;
    warp_inclusive_dual(sm, sf, lane);
    const int32_t m_excl = sm - m_lane0;
    const int32_t f_excl = sf - f_lane0;

    if (lane == WARP_SIZE - 1) {
        warp_tot_m[warp] = sm;
        warp_tot_f[warp] = sf;
    }
    asc_syncthreads();

    if (tid == 0) {
        int32_t rm = 0;
        int32_t rf = 0;
        for (int32_t w = 0; w < NWARPS; ++w) {
            warp_pref_m[w] = rm;
            warp_pref_f[w] = rf;
            rm += warp_tot_m[w];
            rf += warp_tot_f[w];
        }
        // 排他首元素 = 核间前缀
        mask_offset[0] = pref_m;
        full_offset[0] = pref_f;
    }
    asc_syncthreads();

    int32_t run_m = pref_m + warp_pref_m[warp] + m_excl;
    int32_t run_f = pref_f + warp_pref_f[warp] + f_excl;
    for (int32_t i = lo; i < hi; ++i) {
        run_m += mask_cnt[i];
        run_f += full_cnt[i];
        mask_offset[i + 1] = run_m;
        full_offset[i + 1] = run_f;
    }
}

extern "C" __global__ __aicore__ void dual_exclusive_scan(GM_ADDR mask_cnt, GM_ADDR full_cnt, GM_ADDR mask_offset,
                                                          GM_ADDR full_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    const int32_t n = tilingData.totalLength;
    const int32_t coreNum = tilingData.coreNum;

    const int32_t coreIdx = static_cast<int32_t>(GetBlockIdx());
    const int32_t per = (n + coreNum - 1) / coreNum;
    const int32_t base = coreIdx * per;
    const int32_t end = (base + per < n) ? (base + per) : n;
    const int32_t len = end - base;

    __gm__ int32_t* base_mask_cnt = (__gm__ int32_t*)mask_cnt + base;
    __gm__ int32_t* base_full_cnt = (__gm__ int32_t*)full_cnt + base;
    __gm__ int32_t* base_mask_off = (__gm__ int32_t*)mask_offset + base;
    __gm__ int32_t* base_full_off = (__gm__ int32_t*)full_offset + base;

    int32_t pref_m = 0;
    int32_t pref_f = 0;
    if (coreNum > 1) {
        __gm__ int32_t* user_ws = (__gm__ int32_t*)GetUserWorkspace(workspace);
        __gm__ int32_t* mask_tot = user_ws;
        __gm__ int32_t* full_tot = user_ws + (coreNum + 1);

        // 阶段一：各核计算本地总和写入 workspace（VF 串序执行，完成后才进入后续代码）
        asc_vf_call<dual_scan_totals_vf>(dim3{SCAN_THREADS}, base_mask_cnt, base_full_cnt, len, mask_tot, full_tot,
                                         coreIdx);

        // 核间同步（SyncAll 自带内存可见性，无需额外 fence）
        AscendC::SyncAll<true>();

        for (int32_t i = 0; i < coreIdx; ++i) {
            pref_m += mask_tot[i];
            pref_f += full_tot[i];
        }
    }

    // 阶段二：单核场景跳过同步直接扫描
    asc_vf_call<dual_scan_final_vf>(dim3{SCAN_THREADS}, base_mask_cnt, base_full_cnt, base_mask_off, base_full_off, len,
                                    pref_m, pref_f);
}
