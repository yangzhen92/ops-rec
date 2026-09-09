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

/**
 * 预计算 kernel（K2Q 的 kv-range 优化前置阶段）：
 * 为每个 (b, h, q_block) 计算其 q_token 有效区间的 min/max kv，写入 q_block_kv_min / q_block_kv_max。
 *
 * - 并行单元为 (b, h, q_block)，totalBlocks = num_q_blocks * H * B。
 * - warp 级模式：每个 warp 处理一个 (b, h, q_block)，warp 内 32 个 lane 各处理
 *   Q_BLOCK_SIZE / 32 个 q_token。
 * - 布局 [B, H, func_q_len, n_func]，区间为 [F_{2i}, F_{2i+1})。
 *
 * @tparam N_FUNC: 函数值个数（编译期常量，0 表示使用运行期 n_func）
 * @tparam Q_BLOCK_SIZE_T: Q 块大小（编译期常量，0 表示使用运行期值）
 */
template <int32_t N_FUNC, int32_t Q_BLOCK_SIZE_T = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void compute_q_block_kv_range_kernel_vf(
    const __gm__ int32_t* func_tensor,  // [B, H, func_q_len, n_func]
    int32_t stride_b, int32_t stride_h, int32_t stride_q, int32_t stride_f, int32_t Q_LEN, int32_t n_func,
    int32_t Q_BLOCK_SIZE_RT, int32_t num_q_blocks,
    __gm__ int32_t* q_block_kv_min,  // [B, H, num_q_blocks]
    __gm__ int32_t* q_block_kv_max,  // [B, H, num_q_blocks]
    int32_t H, int32_t B)
{
    const int32_t total_blocks = num_q_blocks * H * B;
    const int32_t Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;

    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;  // [0..WARP_LEVEL_NUM-1]
    const int32_t lane_id = tid % WARP_SIZE_CONST;  // [0..31]

    // Grid-stride 循环：每个 pass 处理 WARP_LEVEL_NUM 个任务（每个 warp 一个）
    const int32_t grid_ceil = DIVUP(total_blocks, WARP_LEVEL_NUM);
    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < grid_ceil; pass += AscendC::Simt::GetBlockNum()) {
        const int32_t my_linear = pass * WARP_LEVEL_NUM + warp_id;
        const int32_t b = my_linear / (num_q_blocks * H);
        const int32_t rem = my_linear - b * (num_q_blocks * H);
        const int32_t h = rem / num_q_blocks;
        const int32_t q_block = rem - h * num_q_blocks;

        // 计算该 q_block 的 q 范围
        const int32_t q_start = q_block * Q_BLOCK_SIZE;
        const int32_t q_end = (my_linear < total_blocks) ? min(q_start + Q_BLOCK_SIZE, Q_LEN) : q_start;
        const int32_t num_active_tokens = q_end - q_start;

        int32_t thread_min_kv = INT_MAX;
        int32_t thread_max_kv = 0;

        if constexpr (Q_BLOCK_SIZE_T > 0) {
            // 编译期块大小：保留 unroll 展开（hot path 不变）TOKENS_PER_THREAD
            constexpr int32_t TPT = Q_BLOCK_SIZE_T / WARP_SIZE_CONST;
#pragma unroll
            for (int32_t t = 0; t < TPT; t++) {
                const int32_t local_q_idx = lane_id * TPT + t;
                if (local_q_idx >= num_active_tokens) {
                    continue;
                }
                const int32_t q_idx = q_start + local_q_idx;
                const __gm__ int32_t* my_func_ptr = func_tensor + b * stride_b + h * stride_h + q_idx * stride_q;
                int32_t token_min_kv = INT_MAX;
                int32_t token_max_kv = 0;
                block_sparse_func::get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, token_min_kv,
                                                              token_max_kv);
                thread_min_kv = min(thread_min_kv, token_min_kv);
                thread_max_kv = max(thread_max_kv, token_max_kv);
            }
        } else {
            // 运行期块大小（含512）：每个 lane 按 warp 步长遍历所有 q_token
            for (int32_t local_q_idx = lane_id; local_q_idx < num_active_tokens; local_q_idx += WARP_SIZE_CONST) {
                const int32_t q_idx = q_start + local_q_idx;
                const __gm__ int32_t* my_func_ptr = func_tensor + b * stride_b + h * stride_h + q_idx * stride_q;
                int32_t token_min_kv = INT_MAX;
                int32_t token_max_kv = 0;
                block_sparse_func::get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, token_min_kv,
                                                              token_max_kv);
                thread_min_kv = min(thread_min_kv, token_min_kv);
                thread_max_kv = max(thread_max_kv, token_max_kv);
            }
        }

        const int32_t warp_min = asc_reduce_min(thread_min_kv);
        const int32_t warp_max = asc_reduce_max(thread_max_kv);

        if (lane_id == 0 && my_linear < total_blocks) {
            const int32_t out_offset = b * H * num_q_blocks + h * num_q_blocks + q_block;
            q_block_kv_min[out_offset] = warp_min;
            q_block_kv_max[out_offset] = warp_max;
        }
    }
}

}  // namespace

// =============================================================================
// extern "C" 入口：依据运行期 nFunc / qBlockSize 做模板分发
// =============================================================================
// 直接调用 VF kernel（通过 asc_vf_call）
#define KVRANGE_LAUNCH_VF(NF, QBS)                                                                                     \
    asc_vf_call<compute_q_block_kv_range_kernel_vf<NF, QBS>>(                                                          \
        dim3{static_cast<uint32_t>(WARP_LEVEL_NUM * WARP_SIZE_CONST)}, (__gm__ int32_t*)func_tensor, strideB, strideH, \
        strideQ, strideF, qLen, nFunc, qBlockSize, numQBlocks, (__gm__ int32_t*)q_block_kv_min,                        \
        (__gm__ int32_t*)q_block_kv_max, hNum, bNum)

// 按 Q_BLOCK_SIZE 分发（对应原始 dynamic_emb 中的 DISPATCH_K2Q_BY_NFUNC_BS）
#define DISPATCH_KVRANGE_BY_BLOCK_SIZE(NF) \
    if (qBlockSize == 512) {               \
        KVRANGE_LAUNCH_VF(NF, 512);        \
    } else if (qBlockSize == 256) {        \
        KVRANGE_LAUNCH_VF(NF, 256);        \
    } else if (qBlockSize == 128) {        \
        KVRANGE_LAUNCH_VF(NF, 128);        \
    } else if (qBlockSize == 64) {         \
        KVRANGE_LAUNCH_VF(NF, 64);         \
    } else {                               \
        KVRANGE_LAUNCH_VF(NF, 0);          \
    }

extern "C" __global__ __aicore__ void compute_q_block_kv_range(GM_ADDR func_tensor, GM_ADDR q_block_kv_min,
                                                               GM_ADDR q_block_kv_max, GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int32_t strideB = tilingData.strideB;
    int32_t strideH = tilingData.strideH;
    int32_t strideQ = tilingData.strideQ;
    int32_t strideF = tilingData.strideF;
    int32_t qLen = tilingData.qLen;
    int32_t nFunc = tilingData.nFunc;
    int32_t qBlockSize = tilingData.qBlockSize;
    int32_t numQBlocks = tilingData.numQBlocks;
    int32_t hNum = tilingData.hNum;
    int32_t bNum = tilingData.bNum;

    switch (nFunc) {
        case 2:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(2);
            break;
        case 4:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(4);
            break;
        case 6:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(6);
            break;
        case 8:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(8);
            break;
        case 10:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(10);
            break;
        case 12:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(12);
            break;
        case 14:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(14);
            break;
        case 16:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(16);
            break;
        case 18:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(18);
            break;
        case 20:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(20);
            break;
        case 22:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(22);
            break;
        case 24:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(24);
            break;
        case 26:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(26);
            break;
        case 28:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(28);
            break;
        case 30:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(30);
            break;
        case 32:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(32);
            break;
        default:
            DISPATCH_KVRANGE_BY_BLOCK_SIZE(0);
            break;
    }
}
