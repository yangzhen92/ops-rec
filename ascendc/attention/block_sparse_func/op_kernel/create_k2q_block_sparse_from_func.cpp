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
constexpr int32_t MAX_NUM_WARPS = 8;
}  // namespace

/**
 * K2Q（反向）：为每个 (b, h, kv_block) 并行计算其覆盖的 q_block 列表。
 * 布局 [B, H, func_q_len, n_func]，n_func 为偶数，区间为 [F_{2i}, F_{2i+1})。
 *
 * @tparam N_FUNC: 函数值个数（编译期常量，0 表示使用运行期 n_func）
 * @tparam Q_BLOCK_SIZE_T: Q 块大小（编译期常量，0 表示使用运行期值）
 * @tparam KV_BLOCK_SIZE_T: KV 块大小（编译期常量，0 表示使用运行期值）
 */
template <int32_t N_FUNC, int32_t Q_BLOCK_SIZE_T = 0, int32_t KV_BLOCK_SIZE_T = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void create_k2q_block_sparse_from_func_kernel_vf(
    const __gm__ int32_t* func_tensor,  // [B, H, func_q_len, n_func]
    int32_t stride_b, int32_t stride_h, int32_t stride_q, int32_t stride_f, int32_t Q_LEN, int32_t KV_LEN,
    int32_t n_func, int32_t Q_BLOCK_SIZE_RT, int32_t KV_BLOCK_SIZE_RT, int32_t num_q_blocks, int32_t num_kv_blocks,
    const __gm__ int32_t* q_block_kv_min,  // [B, H, num_q_blocks] 预计算（可为 nullptr）
    const __gm__ int32_t* q_block_kv_max,  // [B, H, num_q_blocks] 预计算（可为 nullptr）
    __gm__ int32_t* mask_block_cnt,        // [B, H, num_kv_blocks]
    __gm__ int32_t* full_block_cnt,        // [B, H, num_kv_blocks]
    __gm__ int32_t* block_id,              // [B, H, num_kv_blocks, num_q_blocks]
    int32_t H, int32_t B)
{
    // gridDim.x <= 65535；当 (b, h, kv_block) 元组数超过 65535 时用 grid-stride 循环补齐。
    const int32_t total_blocks = num_kv_blocks * H * B;
    const int32_t Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;
    const int32_t KV_BLOCK_SIZE = (KV_BLOCK_SIZE_T > 0) ? KV_BLOCK_SIZE_T : KV_BLOCK_SIZE_RT;

#ifdef ENABLE_WARP_LEVEL_OPT
    constexpr bool USE_WARP_LEVEL_K2Q = (Q_BLOCK_SIZE_T > 0) && (Q_BLOCK_SIZE_T % WARP_SIZE_CONST == 0);
#else
    constexpr bool USE_WARP_LEVEL_K2Q = false;
#endif
    constexpr int32_t TOKENS_PER_THREAD_K2Q = USE_WARP_LEVEL_K2Q ? (Q_BLOCK_SIZE_T / WARP_SIZE_CONST) : 1;

    // 共享内存：块级归约使用（warp 级模式分配 1 个即可）
    __ubuf__ int32_t warp_results_k2q[USE_WARP_LEVEL_K2Q ? 1 : MAX_NUM_WARPS];

    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;  // [0..WARP_LEVEL_NUM-1]
    const int32_t lane_id = tid % WARP_SIZE_CONST;  // [0..31]

    // 寄存器缓存 func 值
    int32_t func_regs_k2q[TOKENS_PER_THREAD_K2Q][N_FUNC > 0 ? N_FUNC : 1];
    int32_t token_active_mask_k2q;  // bit t = token t 是否有效

    const int32_t out_cnt_stride_b = H * num_kv_blocks;
    const int32_t out_cnt_stride_h = num_kv_blocks;
    const int32_t out_idx_stride_b = H * num_kv_blocks * num_q_blocks;
    const int32_t out_idx_stride_h = num_kv_blocks * num_q_blocks;
    const int32_t out_idx_stride_kv = num_q_blocks;

    const int32_t grid_ceil_k2q = DIVUP(total_blocks, USE_WARP_LEVEL_K2Q ? WARP_LEVEL_NUM : 1);
    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < grid_ceil_k2q; pass += AscendC::Simt::GetBlockNum()) {
        int32_t my_linear;
        if constexpr (USE_WARP_LEVEL_K2Q) {
            my_linear = pass * WARP_LEVEL_NUM + warp_id;
        } else {
            my_linear = pass;
        }
        const int32_t b = my_linear / (num_kv_blocks * H);
        const int32_t rem = my_linear - b * (num_kv_blocks * H);
        const int32_t h = rem / num_kv_blocks;
        const int32_t kv_block = rem - h * num_kv_blocks;

        // 计算该 kv_block 的 kv 范围
        const int32_t kv_start = kv_block * KV_BLOCK_SIZE;
        const int32_t kv_end = (my_linear < total_blocks) ? min(kv_start + KV_BLOCK_SIZE, KV_LEN) : kv_start;
        const bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);

        const __gm__ int32_t* base_func_ptr = func_tensor + b * stride_b + h * stride_h;

        const int32_t cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + kv_block;
        const int32_t idx_base = b * out_idx_stride_b + h * out_idx_stride_h + kv_block * out_idx_stride_kv;

        // 预计算的 q_block kv 范围偏移
        const int32_t precompute_base = b * H * num_q_blocks + h * num_q_blocks;
#ifndef DISABLE_KV_RANGE_OPT
        constexpr bool use_precompute = true;
#else
        constexpr bool use_precompute = false;
#endif

        int32_t mask_count = 0;
        int32_t full_count = 0;

        // 遍历所有 q_block
        for (int32_t q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
            // =========================================================================
            // 优化：跳过 kv 范围与当前 kv_block 无交叠的 q_block
            // =========================================================================
            if constexpr (use_precompute) {
                const int32_t q_min_kv = q_block_kv_min[precompute_base + q_block_idx];
                const int32_t q_max_kv = q_block_kv_max[precompute_base + q_block_idx];
                if (q_max_kv <= kv_start || q_min_kv >= kv_end) {
                    continue;
                }
            }

            // 计算该 q_block 的 q 范围
            const int32_t q_start = q_block_idx * Q_BLOCK_SIZE;
            const int32_t q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
            const int32_t num_active_tokens = q_end - q_start;
            const bool is_q_block_full = (num_active_tokens == Q_BLOCK_SIZE);

            // =====================================================================
            // 加载 func 值并计算状态
            // =====================================================================
            uint32_t local_state_mask = 0;
            token_active_mask_k2q = 0;

            // 加载 func 值到寄存器
#pragma unroll
            for (int32_t t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
                int32_t local_q_idx;
                if constexpr (USE_WARP_LEVEL_K2Q) {
                    local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
                } else {
                    local_q_idx = tid;
                }
                const int32_t q_idx = q_start + local_q_idx;

                if (local_q_idx < num_active_tokens) {
                    token_active_mask_k2q |= (1u << t);
                    const __gm__ int32_t* my_func_ptr = base_func_ptr + q_idx * stride_q;
                    if constexpr (N_FUNC > 0) {
                        if (stride_f == 1) {
                            constexpr bool NUM_4 = N_FUNC % 4 == 0;
                            if constexpr (NUM_4) {
                                constexpr int32_t NUM_PAIRS = N_FUNC / 4;
#pragma unroll
                                for (int32_t i = 0; i < NUM_PAIRS; i++) {
                                    const __gm__ int4* func_ptr2 =
                                        reinterpret_cast<const __gm__ int4*>(my_func_ptr + 4 * i);
                                    func_regs_k2q[t][4 * i] = (*func_ptr2).x;
                                    func_regs_k2q[t][4 * i + 1] = (*func_ptr2).y;
                                    func_regs_k2q[t][4 * i + 2] = (*func_ptr2).z;
                                    func_regs_k2q[t][4 * i + 3] = (*func_ptr2).w;
                                }
                            } else {
                                constexpr int32_t NUM_PAIRS = N_FUNC / 2;
#pragma unroll
                                for (int32_t i = 0; i < NUM_PAIRS; i++) {
                                    const __gm__ int2* func_ptr2 =
                                        reinterpret_cast<const __gm__ int2*>(my_func_ptr + 2 * i);
                                    func_regs_k2q[t][2 * i] = (*func_ptr2).x;
                                    func_regs_k2q[t][2 * i + 1] = (*func_ptr2).y;
                                }
                            }
                        } else {
#pragma unroll
                            for (int32_t f = 0; f < N_FUNC; f++) {
                                func_regs_k2q[t][f] = my_func_ptr[f * stride_f];
                            }
                        }
                    }
                }
            }

            // 计算每个 token 的状态
#pragma unroll
            for (int32_t t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
                int32_t token_state;
                if (token_active_mask_k2q & (1 << t)) {
                    if constexpr (N_FUNC > 0) {
                        token_state = block_sparse_func::get_token_kv_block_state_from_regs<N_FUNC / 2>(
                            func_regs_k2q[t], kv_start, kv_end);
                    } else {
                        int32_t local_q_idx;
                        if constexpr (USE_WARP_LEVEL_K2Q) {
                            local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
                        } else {
                            local_q_idx = tid;
                        }
                        const __gm__ int32_t* ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                        token_state = block_sparse_func::get_token_kv_block_state<N_FUNC>(ptr, n_func, stride_f,
                                                                                          kv_start, kv_end);
                    }
                } else {
                    // K2Q 始终进行 q_boundary 检查：inactive（padding）token 贡献 STATE_EMPTY。
                    token_state = STATE_EMPTY;
                }
                local_state_mask |= (1u << token_state);
            }

            // =====================================================================
            // 归约 q_block 的状态
            // =====================================================================
            int32_t q_block_state;
            if constexpr (USE_WARP_LEVEL_K2Q) {
                q_block_state = block_sparse_func::warp_reduce_state_only(local_state_mask);
            } else {
                q_block_state =
                    block_sparse_func::block_reduce_state_bitmask(local_state_mask, warp_results_k2q, warp_id, lane_id);
            }

            // 记录结果
            if (USE_WARP_LEVEL_K2Q ? lane_id == 0 : tid == 0) {
                const bool can_be_full = is_q_block_full && is_kv_block_full;
                const bool is_full_state = (q_block_state == STATE_FULL);
                const bool is_mask_or_full = (q_block_state == STATE_FULL || q_block_state == STATE_MASK);
                if (is_full_state && can_be_full) {
                    block_id[idx_base + full_count] = q_block_idx;
                    full_count++;
                } else if (is_mask_or_full) {
                    block_id[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                    mask_count++;
                }
            }

            if constexpr (!USE_WARP_LEVEL_K2Q) {
                asc_syncthreads();
            }
        }

        // 写最终计数
        const int32_t final_leader_id_k2q = USE_WARP_LEVEL_K2Q ? lane_id : tid;
        if (final_leader_id_k2q == 0 && my_linear < total_blocks) {
            mask_block_cnt[cnt_offset] = mask_count;
            full_block_cnt[cnt_offset] = full_count;
        }
    }
}

// =============================================================================
// extern "C" 入口：依据运行期 nFunc / qBlockSize / kvBlockSize 做模板分发
// =============================================================================
#define K2Q_LAUNCH(NF, QBS, KVBS, BS)                                                                            \
    asc_vf_call<create_k2q_block_sparse_from_func_kernel_vf<NF, QBS, KVBS>>(                                     \
        dim3{static_cast<uint32_t>(BS)}, (__gm__ int32_t*)func_tensor, strideB, strideH, strideQ, strideF, qLen, \
        kvLen, nFunc, qBlockSize, kvBlockSize, numQBlocks, numKvBlocks, (__gm__ int32_t*)q_block_kv_min,         \
        (__gm__ int32_t*)q_block_kv_max, (__gm__ int32_t*)mask_block_cnt, (__gm__ int32_t*)full_block_cnt,       \
        (__gm__ int32_t*)block_idx, hNum, bNum)

#define DISPATCH_K2Q_BY_NFUNC(NF, QBS, KVBS, BS) \
    switch (nFunc) {                             \
        case 2:                                  \
            K2Q_LAUNCH(2, QBS, KVBS, BS);        \
            break;                               \
        case 4:                                  \
            K2Q_LAUNCH(4, QBS, KVBS, BS);        \
            break;                               \
        case 6:                                  \
            K2Q_LAUNCH(6, QBS, KVBS, BS);        \
            break;                               \
        case 8:                                  \
            K2Q_LAUNCH(8, QBS, KVBS, BS);        \
            break;                               \
        case 10:                                 \
            K2Q_LAUNCH(10, QBS, KVBS, BS);       \
            break;                               \
        case 12:                                 \
            K2Q_LAUNCH(12, QBS, KVBS, BS);       \
            break;                               \
        case 14:                                 \
            K2Q_LAUNCH(14, QBS, KVBS, BS);       \
            break;                               \
        case 16:                                 \
            K2Q_LAUNCH(16, QBS, KVBS, BS);       \
            break;                               \
        case 18:                                 \
            K2Q_LAUNCH(18, QBS, KVBS, BS);       \
            break;                               \
        case 20:                                 \
            K2Q_LAUNCH(20, QBS, KVBS, BS);       \
            break;                               \
        case 22:                                 \
            K2Q_LAUNCH(22, QBS, KVBS, BS);       \
            break;                               \
        case 24:                                 \
            K2Q_LAUNCH(24, QBS, KVBS, BS);       \
            break;                               \
        case 26:                                 \
            K2Q_LAUNCH(26, QBS, KVBS, BS);       \
            break;                               \
        case 28:                                 \
            K2Q_LAUNCH(28, QBS, KVBS, BS);       \
            break;                               \
        case 30:                                 \
            K2Q_LAUNCH(30, QBS, KVBS, BS);       \
            break;                               \
        case 32:                                 \
            K2Q_LAUNCH(32, QBS, KVBS, BS);       \
            break;                               \
        default:                                 \
            K2Q_LAUNCH(0, QBS, KVBS, BS);        \
            break;                               \
    }

extern "C" __global__ __aicore__ void create_k2q_block_sparse_from_func(GM_ADDR func_tensor, GM_ADDR q_block_kv_min,
                                                                        GM_ADDR q_block_kv_max, GM_ADDR mask_block_cnt,
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

#ifdef ENABLE_WARP_LEVEL_OPT
    int32_t blockSize = WARP_LEVEL_NUM * WARP_SIZE_CONST;
    if (qBlockSize == 128 && kvBlockSize == 128) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 128, 128, blockSize);
    } else if (qBlockSize == 256 && kvBlockSize == 128) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 256, 128, blockSize);
    } else if (qBlockSize == 64 && kvBlockSize == 64) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 64, 64, blockSize);
    } else if (qBlockSize == 64 && kvBlockSize == 128) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 64, 128, blockSize);
    } else if (qBlockSize == 128 && kvBlockSize == 64) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 128, 64, blockSize);
    } else {
        blockSize = qBlockSize;
        DISPATCH_K2Q_BY_NFUNC(nFunc, 0, 0, qBlockSize);
    }
#else
    int32_t blockSize = qBlockSize;
    if (qBlockSize == 128 && kvBlockSize == 128) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 128, 128, blockSize);
    } else if (qBlockSize == 256 && kvBlockSize == 128) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 256, 128, blockSize);
    } else if (qBlockSize == 64 && kvBlockSize == 64) {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 64, 64, blockSize);
    } else {
        DISPATCH_K2Q_BY_NFUNC(nFunc, 0, 0, blockSize);
    }
#endif  // ENABLE_WARP_LEVEL_OPT
}
