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

#ifndef OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_SIMT_H
#define OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_SIMT_H

#include "common.h"

namespace block_sparse_func {

__forceinline__ __simt_callee__ uint32_t reduce_or_sync(uint32_t value)
{
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value |= asc_shfl_down(value, offset);
    }
    return asc_shfl(value, 0);
}

/**
 * 将合并后的状态位掩码转换为状态值。
 * 位掩码编码：EMPTY(0)->0b001, MASK(1)->0b010, FULL(2)->0b100
 */
__forceinline__ __simt_callee__ int32_t bitmask_to_state(uint32_t mask)
{
    if (mask == 1u)
        return STATE_EMPTY;
    if (mask == 4u)
        return STATE_FULL;
    return STATE_MASK;
}

/**
 * warp 级状态位掩码归约（ENABLE_WARP_LEVEL_OPT 时使用）。
 */
__forceinline__ __simt_callee__ int32_t warp_reduce_state_only(uint32_t local_state_mask)
{
    uint32_t final_mask = reduce_or_sync(local_state_mask);
    return bitmask_to_state(final_mask);
}

/**
 * 块级状态位掩码归约（warp 归约 + 跨 warp 归约，未启用 warp 级优化时使用）。
 */
__forceinline__ __simt_callee__ int32_t block_reduce_state_bitmask(uint32_t local_state_mask,
                                                                   __ubuf__ int32_t* warp_results,
                                                                   const int32_t& warp_id, const int32_t& lane_id)
{
    uint32_t warp_mask = reduce_or_sync(local_state_mask);
    if (lane_id == 0) {
        warp_results[warp_id] = static_cast<int32_t>(warp_mask);
    }
    asc_syncthreads();
    uint32_t final_mask = 0;
    if (warp_id == 0) {
        int32_t num_warps = (AscendC::Simt::GetThreadNum() + WARP_SIZE_CONST - 1) / WARP_SIZE_CONST;
        uint32_t my_result = (lane_id < num_warps) ? static_cast<uint32_t>(warp_results[lane_id]) : 0u;
        final_mask = reduce_or_sync(my_result);
    }
    return bitmask_to_state(final_mask);
}

/**
 * 直接读取 func_tensor 计算 q_token 有效区间的 min/max kv 位置。
 */
template <int32_t N_FUNC>
__forceinline__ __simt_callee__ void get_token_kv_range(const __gm__ int32_t* func_ptr, int32_t n_func,
                                                        int32_t stride_f, int32_t& out_min_kv, int32_t& out_max_kv)
{
    const int32_t num_intervals = (N_FUNC > 0) ? (N_FUNC / 2) : (n_func / 2);
    int32_t min_kv = INT_MAX;
    int32_t max_kv = 0;
#pragma unroll
    for (int32_t i = 0; i < num_intervals; i++) {
        int32_t interval_start, interval_end;
        if (stride_f == 1) {
            __gm__ int2* func_ptr2 = reinterpret_cast<__gm__ int2*>(const_cast<__gm__ int32_t*>(func_ptr + 2 * i));
            interval_start = (*func_ptr2).x;
            interval_end = (*func_ptr2).y;
        } else {
            interval_start = func_ptr[(2 * i) * stride_f];
            interval_end = func_ptr[(2 * i + 1) * stride_f];
        }
        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }
    out_min_kv = min_kv;
    out_max_kv = max_kv;
}

/**
 * 使用寄存器缓存的 func 值计算 q_token 有效区间的 min/max kv 位置。
 * 布局 [B, H, func_q_len, n_func]，n_func 为偶数，区间为 [F_{2i}, F_{2i+1})。
 */
template <int32_t N_FUNC>
__forceinline__ __simt_callee__ void get_token_kv_range_from_regs(const int32_t* func_regs, int32_t& out_min_kv,
                                                                  int32_t& out_max_kv)
{
    constexpr int32_t num_intervals = N_FUNC / 2;
    int32_t min_kv = INT_MAX;
    int32_t max_kv = 0;
#pragma unroll
    for (int32_t i = 0; i < num_intervals; i++) {
        int32_t interval_start = func_regs[2 * i];
        int32_t interval_end = func_regs[2 * i + 1];
        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }
    out_min_kv = min_kv;
    out_max_kv = max_kv;
}

/**
 * 判断 (q_token, kv_block) 的状态：FULL / MASK / EMPTY。
 */
template <int32_t N_FUNC>
__forceinline__ __simt_callee__ int32_t get_token_kv_block_state(const __gm__ int32_t* func_ptr, int32_t n_func,
                                                                 int32_t stride_f, int32_t kv_start, int32_t kv_end)
{
    const int32_t num_intervals = (N_FUNC > 0) ? (N_FUNC / 2) : (n_func / 2);
    bool has_any_overlap = false;
    bool is_fully_covered = false;
#pragma unroll
    for (int32_t i = 0; i < num_intervals; i++) {
        int32_t interval_start, interval_end;
        if (stride_f == 1) {
            __gm__ int2* func_ptr2 = reinterpret_cast<__gm__ int2*>(const_cast<__gm__ int32_t*>(func_ptr + 2 * i));
            interval_start = (*func_ptr2).x;
            interval_end = (*func_ptr2).y;
        } else {
            interval_start = func_ptr[(2 * i) * stride_f];
            interval_end = func_ptr[(2 * i + 1) * stride_f];
        }
        if (interval_end <= interval_start) {
            continue;
        }
        if (interval_start <= kv_start && interval_end >= kv_end) {
            is_fully_covered = true;
        } else if (interval_start < kv_end && interval_end > kv_start) {
            has_any_overlap = true;
        }
    }
    if (is_fully_covered) {
        return STATE_FULL;
    } else if (has_any_overlap) {
        return STATE_MASK;
    }
    return STATE_EMPTY;
}

/**
 * 使用寄存器缓存的 func 值判断 (q_token, kv_block) 状态。
 * @tparam N_INTERVALS: 区间数（N_FUNC / 2）
 */
template <int32_t N_INTERVALS>
__forceinline__ __simt_callee__ int32_t get_token_kv_block_state_from_regs(const int32_t* func_regs, int32_t kv_start,
                                                                           int32_t kv_end)
{
    bool has_any_overlap = false;
    bool is_fully_covered = false;
#pragma unroll
    for (int32_t i = 0; i < N_INTERVALS; i++) {
        int32_t interval_start = func_regs[2 * i];
        int32_t interval_end = func_regs[2 * i + 1];
        if (interval_end <= interval_start) {
            continue;
        }
        if (interval_start <= kv_start && interval_end >= kv_end) {
            is_fully_covered = true;
        } else if (interval_start < kv_end && interval_end > kv_start) {
            has_any_overlap = true;
        }
    }
    if (is_fully_covered) {
        return STATE_FULL;
    } else if (has_any_overlap) {
        return STATE_MASK;
    }
    return STATE_EMPTY;
}

}  // namespace block_sparse_func

#endif  // OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_SIMT_H
