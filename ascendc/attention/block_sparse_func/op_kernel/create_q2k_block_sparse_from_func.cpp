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
#include <simt_api/device_warp_functions.h>
#include <simt_api/device_functions.h>
#include "kernel_operator.h"
#include "block_sparse_common.h"
#include "block_sparse_common_simt.h"

using namespace AscendC;

namespace {
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
constexpr int32_t MAX_Q_BLOCK_SIZE = 128;
constexpr int32_t MAX_NUM_WARPS = (MAX_Q_BLOCK_SIZE / WARP_SIZE_CONST);

/**
 * 块级归约：计算需要遍历的 kv_block 范围。
 * 每个线程提供其 q_token 的 min/max kv，块级归约得到整体 min/max，再换算为 kv_block 范围。
 */
__forceinline__ __simt_callee__ void reduce_kv_block_range(
    int32_t thread_min_kv, int32_t thread_max_kv, int32_t KV_BLOCK_SIZE, int32_t num_kv_blocks,
    __ubuf__ int32_t* warp_min_results,  // shared memory [MAX_NUM_WARPS]
    __ubuf__ int32_t* warp_max_results,  // shared memory [MAX_NUM_WARPS]
    int32_t& out_kv_block_start, int32_t& out_kv_block_end)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t lane_id = tid % WARP_SIZE_CONST;
    const int32_t warp_id = tid / WARP_SIZE_CONST;
    const int32_t num_warps = (AscendC::Simt::GetThreadNum() + WARP_SIZE_CONST - 1) / WARP_SIZE_CONST;

    // warp 级 min/max 归约
    int32_t warp_min = asc_reduce_min(thread_min_kv);
    int32_t warp_max = asc_reduce_max(thread_max_kv);
    if (lane_id == 0) {
        warp_min_results[warp_id] = warp_min;
        warp_max_results[warp_id] = warp_max;
    }
    asc_syncthreads();

    int32_t kv_block_start = 0;
    int32_t kv_block_end = 0;
    if (warp_id == 0) {
        int32_t my_min = (lane_id < num_warps) ? warp_min_results[lane_id] : INT_MAX;
        int32_t my_max = (lane_id < num_warps) ? warp_max_results[lane_id] : 0;
        int32_t block_min_kv = asc_reduce_min(my_min);
        int32_t block_max_kv = asc_reduce_max(my_max);
        if (block_max_kv > 0 && block_min_kv != INT_MAX) {
            kv_block_start = block_min_kv / KV_BLOCK_SIZE;
            kv_block_end = min(DIVUP(block_max_kv, KV_BLOCK_SIZE), num_kv_blocks);
        }
    }

    if (tid == 0) {
        warp_min_results[0] = kv_block_start;
        warp_max_results[0] = kv_block_end;
    }
    asc_syncthreads();

    out_kv_block_start = warp_min_results[0];
    out_kv_block_end = warp_max_results[0];
}
}  // namespace

/**
 * Q2K（正向）：为每个 (b, h, q_block) 并行计算其覆盖的 kv_block 列表。
 * 布局 [B, H, func_q_len, n_func]，n_func 为偶数，区间为 [F_{2i}, F_{2i+1})。
 *
 * @tparam N_FUNC: 函数值个数（编译期常量，0 表示使用运行期 n_func）
 * @tparam Q_BLOCK_SIZE_T: Q 块大小（编译期常量，0 表示使用运行期值）
 * @tparam KV_BLOCK_SIZE_T: KV 块大小（编译期常量，0 表示使用运行期值）
 */
template <int32_t N_FUNC, int32_t Q_BLOCK_SIZE_T = 0, int32_t KV_BLOCK_SIZE_T = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void create_q2k_block_sparse_from_func_kernel_vf(
    const __gm__ int32_t* func_tensor,  // [B, H, func_q_len, n_func]
    int32_t stride_b, int32_t stride_h, int32_t stride_q, int32_t stride_f, int32_t Q_LEN, int32_t KV_LEN,
    int32_t n_func, int32_t Q_BLOCK_SIZE_RT, int32_t KV_BLOCK_SIZE_RT, int32_t num_q_blocks, int32_t num_kv_blocks,
    bool check_q_boundary, __gm__ int32_t* mask_block_cnt, __gm__ int32_t* full_block_cnt, __gm__ int32_t* block_id,
    int32_t hNum, int32_t bNum)
{
    const int32_t total_blocks = num_q_blocks * hNum * bNum;
    const int32_t Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;
    const int32_t KV_BLOCK_SIZE = (KV_BLOCK_SIZE_T > 0) ? KV_BLOCK_SIZE_T : KV_BLOCK_SIZE_RT;

#ifdef ENABLE_WARP_LEVEL_OPT
    constexpr bool USE_WARP_LEVEL = (Q_BLOCK_SIZE_T > 0) && (Q_BLOCK_SIZE_T % WARP_SIZE_CONST == 0);
#else
    constexpr bool USE_WARP_LEVEL = false;
#endif
    constexpr int32_t TOKENS_PER_THREAD = USE_WARP_LEVEL ? (Q_BLOCK_SIZE_T / WARP_SIZE_CONST) : 1;

    // 共享内存：块级归约使用（warp 级模式分配 1 个即可）
    __ubuf__ int32_t warp_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];
    __ubuf__ int32_t warp_min_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];
    __ubuf__ int32_t warp_max_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];

    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;
    const int32_t lane_id = tid % WARP_SIZE_CONST;

    // 寄存器缓存 func 值
    int32_t func_regs[TOKENS_PER_THREAD][N_FUNC > 0 ? N_FUNC : 1];
    int32_t token_active_mask;

    const int32_t out_cnt_stride_b = hNum * num_q_blocks;
    const int32_t out_cnt_stride_h = num_q_blocks;
    const int32_t out_idx_stride_b = hNum * num_q_blocks * num_kv_blocks;
    const int32_t out_idx_stride_h = num_q_blocks * num_kv_blocks;
    const int32_t out_idx_stride_q = num_kv_blocks;

    // warp 级：每个 warp 处理不同 (b, h, q_block)；块级：整块处理同一 (b, h, q_block)
    const int32_t grid_ceil = DIVUP(total_blocks, USE_WARP_LEVEL ? WARP_LEVEL_NUM : 1);
    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < grid_ceil; pass += AscendC::Simt::GetBlockNum()) {
        int32_t my_linear;
        if constexpr (USE_WARP_LEVEL) {
            my_linear = pass * WARP_LEVEL_NUM + warp_id;
        } else {
            my_linear = pass;
        }
        const int32_t b = my_linear / (num_q_blocks * hNum);
        const int32_t rem = my_linear - b * (num_q_blocks * hNum);
        const int32_t h = rem / num_q_blocks;
        const int32_t q_block = rem - h * num_q_blocks;

        const int32_t q_start = q_block * Q_BLOCK_SIZE;
        const int32_t q_end = (my_linear < total_blocks) ? min(q_start + Q_BLOCK_SIZE, Q_LEN) : q_start;
        const int32_t num_active_tokens = q_end - q_start;
        const bool is_q_block_full = (num_active_tokens == Q_BLOCK_SIZE);

        const __gm__ int32_t* base_func_ptr = func_tensor + b * stride_b + h * stride_h;

        const int32_t q_idx_primary = q_start + tid;
        const bool is_active_primary = (tid < num_active_tokens);
        const __gm__ int32_t* my_func_ptr = base_func_ptr + q_idx_primary * stride_q;

#ifndef DISABLE_REG_CACHE
        // 加载 func 值到寄存器
        token_active_mask = 0;
#pragma unroll
        for (int32_t t = 0; t < TOKENS_PER_THREAD; t++) {
            int32_t local_q_idx;
            if constexpr (USE_WARP_LEVEL) {
                local_q_idx = lane_id * TOKENS_PER_THREAD + t;
            } else {
                local_q_idx = tid;
            }
            int32_t q_idx = q_start + local_q_idx;
            if (local_q_idx < num_active_tokens) {
                token_active_mask |= (1 << t);
                __gm__ int32_t* func_ptr = const_cast<__gm__ int32_t*>(base_func_ptr) + q_idx * stride_q;
                if constexpr (N_FUNC > 0) {
                    if (stride_f == 1) {
                        constexpr int32_t NUM_PAIRS = N_FUNC / 2;
#pragma unroll
                        for (int32_t i = 0; i < NUM_PAIRS; i++) {
                            __gm__ int2* func_ptr2 = reinterpret_cast<__gm__ int2*>(func_ptr + 2 * i);
                            func_regs[t][2 * i] = (*func_ptr2).x;
                            func_regs[t][2 * i + 1] = (*func_ptr2).y;
                        }
                    } else {
#pragma unroll
                        for (int32_t f = 0; f < N_FUNC; f++) {
                            func_regs[t][f] = func_ptr[f * stride_f];
                        }
                    }
                }
            }
        }
#else
        token_active_mask = is_active_primary ? 1 : 0;
#endif  // DISABLE_REG_CACHE

        // 计算 kv_block 遍历范围
        int32_t kv_block_start, kv_block_end;
#ifdef DISABLE_KV_RANGE_OPT
        kv_block_start = 0;
        kv_block_end = num_kv_blocks;
#else
        {
            int32_t thread_min_kv = INT_MAX;
            int32_t thread_max_kv = 0;
            if constexpr (USE_WARP_LEVEL) {
#pragma unroll
                for (int32_t t = 0; t < TOKENS_PER_THREAD; t++) {
                    if (token_active_mask & (1 << t)) {
                        int32_t token_min_kv, token_max_kv;
                        block_sparse_func::get_token_kv_range_from_regs<N_FUNC>(func_regs[t], token_min_kv,
                                                                                token_max_kv);
                        thread_min_kv = min(thread_min_kv, token_min_kv);
                        thread_max_kv = max(thread_max_kv, token_max_kv);
                    }
                }
                int32_t block_min_kv = asc_reduce_min(thread_min_kv);
                int32_t block_max_kv = asc_reduce_max(thread_max_kv);
                if (block_max_kv > 0 && block_min_kv != INT_MAX) {
                    kv_block_start = block_min_kv / KV_BLOCK_SIZE;
                    kv_block_end = min(DIVUP(block_max_kv, KV_BLOCK_SIZE), num_kv_blocks);
                } else {
                    kv_block_start = 0;
                    kv_block_end = 0;
                }
            } else {
                if (is_active_primary) {
#ifndef DISABLE_REG_CACHE
                    if constexpr (N_FUNC > 0) {
                        block_sparse_func::get_token_kv_range_from_regs<N_FUNC>(func_regs[0], thread_min_kv,
                                                                                thread_max_kv);
                    } else {
                        block_sparse_func::get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, thread_min_kv,
                                                                      thread_max_kv);
                    }
#else
                    block_sparse_func::get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, thread_min_kv,
                                                                  thread_max_kv);
#endif
                }
                reduce_kv_block_range(thread_min_kv, thread_max_kv, KV_BLOCK_SIZE, num_kv_blocks, warp_min_results,
                                      warp_max_results, kv_block_start, kv_block_end);
            }
        }
#endif  // DISABLE_KV_RANGE_OPT

        const int32_t cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + q_block;
        const int32_t idx_base = b * out_idx_stride_b + h * out_idx_stride_h + q_block * out_idx_stride_q;

        int32_t mask_count = 0;
        int32_t full_count = 0;
        constexpr int32_t leader_id = 0;

        for (int32_t kv_block = kv_block_start; kv_block < kv_block_end; kv_block++) {
            const int32_t kv_start = kv_block * KV_BLOCK_SIZE;
            const int32_t kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
            const bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);

            uint32_t local_state_mask = 0;
#pragma unroll
            for (int32_t t = 0; t < TOKENS_PER_THREAD; t++) {
                int32_t token_state;
                int32_t active_state = STATE_EMPTY;
                int32_t inactive_state = check_q_boundary ? STATE_EMPTY : STATE_FULL;

#ifndef DISABLE_REG_CACHE
                bool is_active = token_active_mask & (1 << t);
                if constexpr (N_FUNC > 0) {
                    active_state = block_sparse_func::get_token_kv_block_state_from_regs<N_FUNC / 2>(func_regs[t],
                                                                                                     kv_start, kv_end);
                } else {
                    int32_t local_q_idx;
                    if constexpr (USE_WARP_LEVEL) {
                        local_q_idx = lane_id * TOKENS_PER_THREAD + t;
                    } else {
                        local_q_idx = tid;
                    }
                    const __gm__ int32_t* ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                    active_state =
                        block_sparse_func::get_token_kv_block_state<N_FUNC>(ptr, n_func, stride_f, kv_start, kv_end);
                }
                token_state = is_active ? active_state : inactive_state;
#else
                bool is_active = is_active_primary;
                if (is_active) {
                    active_state = block_sparse_func::get_token_kv_block_state<N_FUNC>(my_func_ptr, n_func, stride_f,
                                                                                       kv_start, kv_end);
                }
                token_state = is_active ? active_state : inactive_state;
#endif
                local_state_mask |= (1u << token_state);
            }

            int32_t kv_block_state;
            if constexpr (USE_WARP_LEVEL) {
                kv_block_state = block_sparse_func::warp_reduce_state_only(local_state_mask);
            } else {
                kv_block_state =
                    block_sparse_func::block_reduce_state_bitmask(local_state_mask, warp_results, warp_id, lane_id);
            }

            if (USE_WARP_LEVEL ? (lane_id == leader_id) : (tid == leader_id)) {
                const bool can_be_full = (!check_q_boundary || is_q_block_full) && is_kv_block_full;
                const bool is_full_state = (kv_block_state == STATE_FULL);
                const bool is_mask_or_full = (kv_block_state == STATE_FULL || kv_block_state == STATE_MASK);
                if (is_full_state && can_be_full) {
                    block_id[idx_base + full_count] = kv_block;
                    full_count++;
                } else if (is_mask_or_full) {
                    block_id[idx_base + (num_kv_blocks - 1 - mask_count)] = kv_block;
                    mask_count++;
                }
            }

            if constexpr (!USE_WARP_LEVEL) {
                asc_syncthreads();
            }
        }

        if (USE_WARP_LEVEL ? (lane_id == leader_id) : (tid == leader_id)) {
            mask_block_cnt[cnt_offset] = mask_count;
            full_block_cnt[cnt_offset] = full_count;
        }

        if constexpr (!USE_WARP_LEVEL) {
            asc_syncthreads();
        }
    }
}

// =============================================================================
// extern "C" 入口：依据运行期 nFunc / qBlockSize / kvBlockSize 做模板分发
// =============================================================================
#define Q2K_LAUNCH(NF, QBS, KVBS, BS)                                                                            \
    asc_vf_call<create_q2k_block_sparse_from_func_kernel_vf<NF, QBS, KVBS>>(                                     \
        dim3{static_cast<uint32_t>(BS)}, (__gm__ int32_t*)func_tensor, strideB, strideH, strideQ, strideF, qLen, \
        kvLen, nFunc, qBlockSize, kvBlockSize, numQBlocks, numKvBlocks, checkQBoundary,                          \
        (__gm__ int32_t*)mask_block_cnt, (__gm__ int32_t*)full_block_cnt, (__gm__ int32_t*)block_idx, hNum, bNum)

#define DISPATCH_Q2K_BY_NFUNC(NF, QBS, KVBS, BS) \
    switch (nFunc) {                             \
        case 2:                                  \
            Q2K_LAUNCH(2, QBS, KVBS, BS);        \
            break;                               \
        case 4:                                  \
            Q2K_LAUNCH(4, QBS, KVBS, BS);        \
            break;                               \
        case 6:                                  \
            Q2K_LAUNCH(6, QBS, KVBS, BS);        \
            break;                               \
        case 8:                                  \
            Q2K_LAUNCH(8, QBS, KVBS, BS);        \
            break;                               \
        case 10:                                 \
            Q2K_LAUNCH(10, QBS, KVBS, BS);       \
            break;                               \
        case 12:                                 \
            Q2K_LAUNCH(12, QBS, KVBS, BS);       \
            break;                               \
        case 14:                                 \
            Q2K_LAUNCH(14, QBS, KVBS, BS);       \
            break;                               \
        case 16:                                 \
            Q2K_LAUNCH(16, QBS, KVBS, BS);       \
            break;                               \
        case 18:                                 \
            Q2K_LAUNCH(18, QBS, KVBS, BS);       \
            break;                               \
        case 20:                                 \
            Q2K_LAUNCH(20, QBS, KVBS, BS);       \
            break;                               \
        case 22:                                 \
            Q2K_LAUNCH(22, QBS, KVBS, BS);       \
            break;                               \
        case 24:                                 \
            Q2K_LAUNCH(24, QBS, KVBS, BS);       \
            break;                               \
        case 26:                                 \
            Q2K_LAUNCH(26, QBS, KVBS, BS);       \
            break;                               \
        case 28:                                 \
            Q2K_LAUNCH(28, QBS, KVBS, BS);       \
            break;                               \
        case 30:                                 \
            Q2K_LAUNCH(30, QBS, KVBS, BS);       \
            break;                               \
        case 32:                                 \
            Q2K_LAUNCH(32, QBS, KVBS, BS);       \
            break;                               \
        default:                                 \
            Q2K_LAUNCH(0, QBS, KVBS, BS);        \
            break;                               \
    }

extern "C" __global__ __aicore__ void create_q2k_block_sparse_from_func(GM_ADDR func_tensor, GM_ADDR mask_block_cnt,
                                                                        GM_ADDR full_block_cnt, GM_ADDR block_idx,
                                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int32_t strideB = tilingData.strideB;
    int32_t strideH = tilingData.strideH;
    int32_t strideQ = tilingData.strideQ;
    int32_t strideF = tilingData.strideF;
    int32_t qLen = tilingData.qLen;
    int32_t kvLen = tilingData.kvLen;
    int32_t nFunc = tilingData.nFunc;
    int32_t qBlockSize = tilingData.qBlockSize;
    int32_t kvBlockSize = tilingData.kvBlockSize;
    int32_t numQBlocks = tilingData.numQBlocks;
    int32_t numKvBlocks = tilingData.numKvBlocks;
    int32_t hNum = tilingData.hNum;
    int32_t bNum = tilingData.bNum;
    bool checkQBoundary = (tilingData.checkQBoundary == 1);

#ifdef ENABLE_WARP_LEVEL_OPT
    int32_t blockSize = WARP_LEVEL_NUM * WARP_SIZE_CONST;
    if (qBlockSize == 256 && kvBlockSize == 128) {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 256, 128, blockSize);
    } else if (qBlockSize == 128 && kvBlockSize == 128) {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 128, 128, blockSize);
    } else {
        blockSize = qBlockSize;
        DISPATCH_Q2K_BY_NFUNC(nFunc, 0, 0, qBlockSize);
    }
#else
    int32_t blockSize = qBlockSize;
    if (qBlockSize == 256 && kvBlockSize == 128) {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 256, 128, blockSize);
    } else if (qBlockSize == 128 && kvBlockSize == 128) {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 128, 128, blockSize);
    } else if (qBlockSize == 64 && kvBlockSize == 64) {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 64, 64, blockSize);
    } else {
        DISPATCH_Q2K_BY_NFUNC(nFunc, 0, 0, blockSize);
    }
#endif  // ENABLE_WARP_LEVEL_OPT
}
