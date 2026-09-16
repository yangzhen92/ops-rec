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

/**
 * N_FUNC 为运行期值时的回退路径：直接从 GM 扫描 token 的 kv 区间（每任务仅扫描一次）。
 */
__forceinline__ __simt_callee__ void get_token_kv_range_from_gm(const __gm__ int32_t* func_ptr, int32_t n_func,
                                                                int32_t stride_f, int32_t& out_min_kv,
                                                                int32_t& out_max_kv)
{
    int32_t min_kv = INT_MAX;
    int32_t max_kv = 0;
    for (int32_t f = 0; f + 1 < n_func; f += 2) {
        const int32_t s = func_ptr[f * stride_f];
        const int32_t e = func_ptr[(f + 1) * stride_f];
        if (e > s) {  // 非空区间
            min_kv = min(min_kv, s);
            max_kv = max(max_kv, e);
        }
    }
    out_min_kv = (min_kv == INT_MAX) ? 0 : min_kv;
    out_max_kv = max_kv;
}

/**
 * 将一个 q token 的 func 值加载到寄存器（stride_f==1 时按 int4/int2 向量化）。
 */
template <int32_t N_FUNC>
__forceinline__ __simt_callee__ void load_func_regs_k2q(const __gm__ int32_t* my_func_ptr, int32_t stride_f,
                                                        int32_t* func_regs)
{
    if constexpr (N_FUNC > 0) {
        if (stride_f == 1) {
            constexpr bool NUM_4 = N_FUNC % 4 == 0;
            if constexpr (NUM_4) {
                constexpr int32_t NUM_QUADS = N_FUNC / 4;
#pragma unroll
                for (int32_t i = 0; i < NUM_QUADS; i++) {
                    const __gm__ int4* func_ptr2 = reinterpret_cast<const __gm__ int4*>(my_func_ptr + 4 * i);
                    func_regs[4 * i] = (*func_ptr2).x;
                    func_regs[4 * i + 1] = (*func_ptr2).y;
                    func_regs[4 * i + 2] = (*func_ptr2).z;
                    func_regs[4 * i + 3] = (*func_ptr2).w;
                }
            } else {
                constexpr int32_t NUM_PAIRS = N_FUNC / 2;
#pragma unroll
                for (int32_t i = 0; i < NUM_PAIRS; i++) {
                    const __gm__ int2* func_ptr2 = reinterpret_cast<const __gm__ int2*>(my_func_ptr + 2 * i);
                    func_regs[2 * i] = (*func_ptr2).x;
                    func_regs[2 * i + 1] = (*func_ptr2).y;
                }
            }
        } else {
#pragma unroll
            for (int32_t f = 0; f < N_FUNC; f++) {
                func_regs[f] = my_func_ptr[f * stride_f];
            }
        }
    }
}

/**
 * 块级归约：计算需要遍历的 kv_block 范围（与 Q2K 的 reduce_kv_block_range 相同模式）。
 */
__forceinline__ __simt_callee__ void reduce_kv_block_range_k2q(
    int32_t thread_min_kv, int32_t thread_max_kv, int32_t KV_BLOCK_SIZE, int32_t num_kv_blocks,
    __ubuf__ int32_t* warp_min_results,  // shared memory [MAX_NUM_WARPS]
    __ubuf__ int32_t* warp_max_results,  // shared memory [MAX_NUM_WARPS]
    int32_t& out_kv_block_start, int32_t& out_kv_block_end)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t lane_id = tid % WARP_SIZE_CONST;
    const int32_t warp_id = tid / WARP_SIZE_CONST;
    const int32_t num_warps = (AscendC::Simt::GetThreadNum() + WARP_SIZE_CONST - 1) / WARP_SIZE_CONST;

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
 * K2Q（反向）两阶段实现，解决旧版 (b, h, kv_block) 任务划分导致的 func_tensor 读放大：
 *
 * 第一阶段（fill）：任务粒度为 (b, h, q_block, kv 列 tile)——行任务不足（大 q_len、小 B*H
 * 中大型用例）时由 Host 传入 kv_cols_per_task > 0，把每行的 kv 列切成多个 tile 倍增
 * warp 并行度（行主序线性化：同一 block 的 8 个 warp 处理同一行的相邻 tile，func 行
 * 读在 DCache 命中）。每个任务仅读一次 func_tensor（warp 级路径寄存器缓存全部 func 值），
 * 把该 (q_block, tile) 内每个 kv_block 的状态写入用户 workspace 的状态矩阵
 * state_matrix[B, H, num_q_blocks, num_kv_blocks]（uint8 状态，行号即 q_block 行号）；
 * tile 内行 kv 范围之外的列写 STATE_EMPTY，各 tile 恰好无重叠覆盖全表。
 *
 * 跨核硬同步（SyncAll）后进入第二阶段（compact）：每个 warp 负责一个 (b*h, kv_group)
 * 列组，按 q_block 升序扫描矩阵列，FULL 自左、MASK 自右有序写入 block_idx，
 * 保证 full 列表递增、mask 列表递减，与参考实现完全一致。
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
    int32_t kv_cols_per_task,  // >0 时 fill 任务按 kv 列 tile 切分
    __gm__ uint8_t* state_matrix,  // [B, H, num_q_blocks, num_kv_blocks] 用户 workspace
    int32_t H, int32_t B)
{
    // kv 列切分（仅两阶段 fill 使用）：任务 = (q_block 行) × num_kv_tiles 个列 tile
    const int32_t num_kv_tiles = (kv_cols_per_task > 0) ? DIVUP(num_kv_blocks, kv_cols_per_task) : 1;
    const int32_t total_row_tasks = num_q_blocks * H * B;
    const int32_t total_tasks = total_row_tasks * num_kv_tiles;
    const int32_t Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;
    const int32_t KV_BLOCK_SIZE = (KV_BLOCK_SIZE_T > 0) ? KV_BLOCK_SIZE_T : KV_BLOCK_SIZE_RT;

#ifdef ENABLE_WARP_LEVEL_OPT
    constexpr bool USE_WARP_LEVEL_K2Q = (Q_BLOCK_SIZE_T > 0) && (Q_BLOCK_SIZE_T % WARP_SIZE_CONST == 0);
#else
    constexpr bool USE_WARP_LEVEL_K2Q = false;
#endif
    constexpr int32_t TOKENS_PER_THREAD_K2Q = USE_WARP_LEVEL_K2Q ? (Q_BLOCK_SIZE_T / WARP_SIZE_CONST) : 1;

    // 共享内存：block 级模式使用（warp 级模式分配 1 个即可）。
    // warp_results 对 block_reduce_state_bitmask 做双缓冲(2 * MAX_NUM_WARPS)，
    // range_results 前/后 MAX_NUM_WARPS 分别作 min/max 归约缓冲。
    __ubuf__ int32_t warp_results_k2q[USE_WARP_LEVEL_K2Q ? 1 : (2 * MAX_NUM_WARPS)];
    __ubuf__ int32_t range_results_k2q[USE_WARP_LEVEL_K2Q ? 1 : (2 * MAX_NUM_WARPS)];

    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;  // [0..WARP_LEVEL_NUM-1]
    const int32_t lane_id = tid % WARP_SIZE_CONST;  // [0..31]

    // 寄存器缓存 func 值（每任务仅从 GM 加载一次）
    int32_t func_regs_k2q[TOKENS_PER_THREAD_K2Q][N_FUNC > 0 ? N_FUNC : 1];
    int32_t token_active_mask_k2q;  // bit t = token t 是否有效

    const int32_t grid_ceil_k2q = DIVUP(total_tasks, USE_WARP_LEVEL_K2Q ? WARP_LEVEL_NUM : 1);
    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < grid_ceil_k2q; pass += AscendC::Simt::GetBlockNum()) {
        int32_t my_linear;
        if constexpr (USE_WARP_LEVEL_K2Q) {
            my_linear = pass * WARP_LEVEL_NUM + warp_id;
        } else {
            my_linear = pass;
        }
        // my_linear = 行号 × num_kv_tiles + tile 号（行主序：相邻 warp 任务同 tile 行相邻，
        // func 行读 DCache 命中）；行号 = (b * H + h) * num_q_blocks + q_block_idx
        if (my_linear >= total_tasks) {
            continue;
        }
        const int32_t row_linear = (num_kv_tiles > 1) ? (my_linear / num_kv_tiles) : my_linear;
        const int32_t kv_tile = (num_kv_tiles > 1) ? (my_linear - row_linear * num_kv_tiles) : 0;
        const int32_t tile_lo = kv_tile * kv_cols_per_task;  // 不切分时恒为 0
        const int32_t tile_hi = (num_kv_tiles > 1) ? min(tile_lo + kv_cols_per_task, num_kv_blocks) : num_kv_blocks;
        const int32_t b = row_linear / (num_q_blocks * H);
        const int32_t rem = row_linear - b * (num_q_blocks * H);
        const int32_t h = rem / num_q_blocks;
        const int32_t q_block_idx = rem - h * num_q_blocks;

        const int32_t q_start = q_block_idx * Q_BLOCK_SIZE;
        const int32_t q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
        const int32_t num_active_tokens = q_end - q_start;

        const __gm__ int32_t* base_func_ptr = func_tensor + b * stride_b + h * stride_h;
        // 状态矩阵行必须按真实行号 row_linear 索引（my_linear 含 tile 号，切分时会越界写串行）
        __gm__ uint8_t* row_ptr = state_matrix + static_cast<int64_t>(row_linear) * num_kv_blocks;

        // =====================================================================
        // 步骤 1：计算该 q_block 覆盖的 kv_block 范围 [kv_block_start, kv_block_end)
        // =====================================================================
        int32_t kv_block_start = 0;
        int32_t kv_block_end = 0;
        if constexpr (USE_WARP_LEVEL_K2Q) {
            // warp 级：加载 func 到寄存器（每任务仅读一次 GM），lane 间归约得到 kv 范围
            token_active_mask_k2q = 0;
            int32_t thread_min_kv = INT_MAX;
            int32_t thread_max_kv = 0;
#pragma unroll
            for (int32_t t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
                const int32_t local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
                if (local_q_idx < num_active_tokens) {
                    token_active_mask_k2q |= (1u << t);
                    const __gm__ int32_t* my_func_ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                    int32_t mn;
                    int32_t mx;
                    if constexpr (N_FUNC > 0) {
                        load_func_regs_k2q<N_FUNC>(my_func_ptr, stride_f, func_regs_k2q[t]);
                        block_sparse_func::get_token_kv_range_from_regs<N_FUNC>(func_regs_k2q[t], mn, mx);
                    } else {
                        get_token_kv_range_from_gm(my_func_ptr, n_func, stride_f, mn, mx);
                    }
                    thread_min_kv = min(thread_min_kv, mn);
                    thread_max_kv = max(thread_max_kv, mx);
                }
            }
            const int32_t warp_min_kv = asc_reduce_min(thread_min_kv);
            const int32_t warp_max_kv = asc_reduce_max(thread_max_kv);
            if (warp_max_kv > 0 && warp_min_kv != INT_MAX) {
                kv_block_start = warp_min_kv / KV_BLOCK_SIZE;
                kv_block_end = min(DIVUP(warp_max_kv, KV_BLOCK_SIZE), num_kv_blocks);
            }
        } else {
            // block 级回退路径：从 GM 逐 token 求区间，块级归约
            const int32_t block_threads = AscendC::Simt::GetThreadNum();
            const int32_t tpt_range = DIVUP(num_active_tokens, block_threads);
            int32_t thread_min_kv = INT_MAX;
            int32_t thread_max_kv = 0;
            for (int32_t tt = 0; tt < tpt_range; tt++) {
                const int32_t local_q_idx = tt * block_threads + tid;
                if (local_q_idx < num_active_tokens) {
                    const __gm__ int32_t* my_func_ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                    int32_t mn;
                    int32_t mx;
                    if constexpr (N_FUNC > 0) {
                        load_func_regs_k2q<N_FUNC>(my_func_ptr, stride_f, func_regs_k2q[0]);
                        block_sparse_func::get_token_kv_range_from_regs<N_FUNC>(func_regs_k2q[0], mn, mx);
                    } else {
                        get_token_kv_range_from_gm(my_func_ptr, n_func, stride_f, mn, mx);
                    }
                    thread_min_kv = min(thread_min_kv, mn);
                    thread_max_kv = max(thread_max_kv, mx);
                }
            }
            reduce_kv_block_range_k2q(thread_min_kv, thread_max_kv, KV_BLOCK_SIZE, num_kv_blocks, range_results_k2q,
                                      range_results_k2q + MAX_NUM_WARPS, kv_block_start, kv_block_end);
        }

        // =====================================================================
        // 步骤 2：tile 内、行 kv 范围外的 kv_block 直接写 STATE_EMPTY
        //（各 tile 无重叠覆盖全行，与步骤 3 的写入范围不重叠）
        // =====================================================================
        if constexpr (USE_WARP_LEVEL_K2Q) {
            for (int32_t i = tile_lo + lane_id; i < tile_hi; i += WARP_SIZE_CONST) {
                if (i < kv_block_start || i >= kv_block_end) {
                    row_ptr[i] = static_cast<uint8_t>(STATE_EMPTY);
                }
            }
        } else {
            const int32_t block_threads = AscendC::Simt::GetThreadNum();
            for (int32_t i = tile_lo + tid; i < tile_hi; i += block_threads) {
                if (i < kv_block_start || i >= kv_block_end) {
                    row_ptr[i] = static_cast<uint8_t>(STATE_EMPTY);
                }
            }
        }

        // =====================================================================
        // 步骤 3：tile 与行 kv 范围交集内逐 kv_block 计算 q_block 状态并写入矩阵行
        // =====================================================================
        const int32_t col_lo = kv_block_start > tile_lo ? kv_block_start : tile_lo;
        const int32_t col_hi = kv_block_end < tile_hi ? kv_block_end : tile_hi;
        int32_t reach_cnt = 0;  // block 级归约双缓冲计数
        for (int32_t kv_block = col_lo; kv_block < col_hi; kv_block++) {
            const int32_t kv_start = kv_block * KV_BLOCK_SIZE;
            const int32_t kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
            uint32_t local_state_mask = 0;
            if constexpr (USE_WARP_LEVEL_K2Q) {
                // warp 级：直接使用寄存器缓存的 func 值
#pragma unroll
                for (int32_t t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
                    int32_t token_state;
                    if (token_active_mask_k2q & (1u << t)) {
                        if constexpr (N_FUNC > 0) {
                            token_state = block_sparse_func::get_token_kv_block_state_from_regs_skip<N_FUNC / 2>(
                                func_regs_k2q[t], kv_start, kv_end);
                        } else {
                            const int32_t local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
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
                const int32_t q_block_state = block_sparse_func::warp_reduce_state_only(local_state_mask);
                if (lane_id == 0) {
                    row_ptr[kv_block] = static_cast<uint8_t>(q_block_state);
                }
            } else {
                // block 级：按 block_threads 步长遍历所负责的 q_token（每 kv 从 GM 重载）
                const int32_t block_threads = AscendC::Simt::GetThreadNum();
                const int32_t tpt = DIVUP(num_active_tokens, block_threads);
                for (int32_t tt = 0; tt < tpt; tt++) {
                    const int32_t local_q_idx = tt * block_threads + tid;
                    int32_t token_state;
                    if (local_q_idx < num_active_tokens) {
                        const __gm__ int32_t* my_func_ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                        if constexpr (N_FUNC > 0) {
                            load_func_regs_k2q<N_FUNC>(my_func_ptr, stride_f, func_regs_k2q[0]);
                            token_state = block_sparse_func::get_token_kv_block_state_from_regs_skip<N_FUNC / 2>(
                                func_regs_k2q[0], kv_start, kv_end);
                        } else {
                            token_state = block_sparse_func::get_token_kv_block_state<N_FUNC>(
                                my_func_ptr, n_func, stride_f, kv_start, kv_end);
                        }
                    } else {
                        // K2Q 始终进行 q_boundary 检查：inactive（padding）token 贡献 STATE_EMPTY。
                        token_state = STATE_EMPTY;
                    }
                    local_state_mask |= (1u << token_state);
                }
                // 双缓冲：相邻两次归约用不同 warp_results 缓冲（block_reduce 内部自带栅栏）
                const int32_t bi = reach_cnt & 1;
                const int32_t q_block_state = block_sparse_func::block_reduce_state_bitmask(
                    local_state_mask, warp_results_k2q + bi * MAX_NUM_WARPS, warp_id, lane_id);
                reach_cnt++;
                if (tid == 0) {
                    row_ptr[kv_block] = static_cast<uint8_t>(q_block_state);
                }
            }
        }
    }
}

/**
 * K2Q 小 shape 单阶段 warp-per-column 路径：
 *
 * 任务粒度为 (b, h, 连续 kv_cols_per_task 列组)，block 内 8 个 warp 以 stride 8
 * 交错各领一列（同一时刻 8 个 warp 读同一 q_block 的 func，DCache/L2 行命中率高）。
 * 每列内按 q_block 升序扫描，warp 级 ballot 归约（无 syncthreads），lane0 直接
 * 有序写输出：full 自左递增、mask 自右（位置递减），天然保序，无需状态矩阵与
 * SyncAll。相对旧版 block-per-column（每列 num_q_blocks 次 block 归约 + 2 倍
 * syncthreads），消除块内栅栏并大幅提升小 shape 并行度：
 *
 * - q_len 小、kv_block_size 小（num_kv_blocks 多）的"多 block 小规模"场景，
 *   Host 按 2 倍核数的目标任务数动态放大 kv_cols_per_task，避免海量超小任务
 *   的调度开销，同时保证满核并行
 * - 代价是 func_tensor 读放大（每列任务全量读 func），仅在读总量可控
 *   （func 小、L2 可缓存，columnTraffic 有预算）时由 Host 侧启发式选中
 */
template <int32_t N_FUNC>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void k2q_single_phase_vf(
    const __gm__ int32_t* func_tensor,  // [B, H, func_q_len, n_func]
    int32_t stride_b, int32_t stride_h, int32_t stride_q, int32_t stride_f, int32_t Q_LEN, int32_t KV_LEN,
    int32_t n_func, int32_t Q_BLOCK_SIZE, int32_t KV_BLOCK_SIZE, int32_t num_q_blocks, int32_t num_kv_blocks,
    int32_t kv_cols_per_task, int32_t H, int32_t B, __gm__ int32_t* mask_block_cnt,  // [B, H, num_kv_blocks]
    __gm__ int32_t* full_block_cnt,  // [B, H, num_kv_blocks]
    __gm__ int32_t* block_id)        // [B, H, num_kv_blocks, num_q_blocks]
{
    constexpr int32_t NUM_WARPS = BLOCK_THREAD_NUM_256 / WARP_SIZE_CONST;  // 8
    const int32_t cols_per_warp = kv_cols_per_task / NUM_WARPS;
    const int32_t num_groups_per_bh = DIVUP(num_kv_blocks, kv_cols_per_task);
    const int32_t total_tasks = num_groups_per_bh * H * B;

    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;
    const int32_t lane_id = tid % WARP_SIZE_CONST;
    int32_t func_regs_sp[N_FUNC > 0 ? N_FUNC : 1];

    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < total_tasks; pass += AscendC::Simt::GetBlockNum()) {
        // pass == (b*H + h)*num_groups_per_bh + col_group
        const int32_t b = pass / (num_groups_per_bh * H);
        const int32_t rem = pass - b * (num_groups_per_bh * H);
        const int32_t h = rem / num_groups_per_bh;
        const int32_t col_group = rem - h * num_groups_per_bh;
        const int32_t col_start = col_group * kv_cols_per_task;

        const __gm__ int32_t* base_func_ptr = func_tensor + b * stride_b + h * stride_h;
        const int32_t cnt_base = (b * H + h) * num_kv_blocks;

        // warp w 领列 col_start + w + c*NUM_WARPS（交错分配，8 warp 同读一段 func）
        for (int32_t c = 0; c < cols_per_warp; c++) {
            const int32_t kv_block = col_start + warp_id + c * NUM_WARPS;
            if (kv_block >= num_kv_blocks) {
                break;
            }

            const int32_t kv_start = kv_block * KV_BLOCK_SIZE;
            const int32_t kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
            const bool is_kv_block_full = (kv_end - kv_start == KV_BLOCK_SIZE);

            const int32_t cnt_offset = cnt_base + kv_block;
            const int32_t idx_base = cnt_offset * num_q_blocks;
            int32_t full_count = 0;
            int32_t mask_count = 0;

            for (int32_t q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
                const int32_t q_start = q_block_idx * Q_BLOCK_SIZE;
                const int32_t q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
                const int32_t num_active_tokens = q_end - q_start;
                uint32_t local_state_mask = 0;
                // warp 内每 lane 交错认领 token：lane + 32*j
                for (int32_t j = 0; j * WARP_SIZE_CONST < num_active_tokens; j++) {
                    const int32_t local_q_idx = j * WARP_SIZE_CONST + lane_id;
                    int32_t token_state;
                    if (local_q_idx < num_active_tokens) {
                        const __gm__ int32_t* my_func_ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                        if constexpr (N_FUNC > 0) {
                            load_func_regs_k2q<N_FUNC>(my_func_ptr, stride_f, func_regs_sp);
                            token_state = block_sparse_func::get_token_kv_block_state_from_regs_skip<N_FUNC / 2>(
                                func_regs_sp, kv_start, kv_end);
                        } else {
                            token_state = block_sparse_func::get_token_kv_block_state<N_FUNC>(
                                my_func_ptr, n_func, stride_f, kv_start, kv_end);
                        }
                    } else {
                        // K2Q 始终进行 q_boundary 检查：inactive（padding）token 贡献 STATE_EMPTY。
                        token_state = STATE_EMPTY;
                    }
                    local_state_mask |= (1u << token_state);
                }
                const int32_t q_block_state = block_sparse_func::warp_reduce_state_only(local_state_mask);
                if (lane_id == 0) {
                    if (q_block_state == STATE_FULL) {
                        if (is_kv_block_full) {
                            block_id[idx_base + full_count] = q_block_idx;
                            full_count++;
                        } else {
                            // 溢出 kv 边界的 kv_block 只能算 mask（与两阶段 compact 语义一致）
                            block_id[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                            mask_count++;
                        }
                    } else if (q_block_state == STATE_MASK) {
                        block_id[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                        mask_count++;
                    }
                }
            }
            if (lane_id == 0) {
                mask_block_cnt[cnt_offset] = mask_count;
                full_block_cnt[cnt_offset] = full_count;
            }
        }
    }
}

/**
 * K2Q 第二阶段（compact）：按列扫描状态矩阵，有序紧缩输出。
 * 每个 warp 负责一个 (b*h, kv_group) 列组，lane 对应组内一个 kv_block；
 * 同 warp 内 32 个 lane 读取矩阵中 32 个连续列，DCache 行命中率高。
 * full 列表自左按 q_block 升序写入，mask 列表自右按 q_block 升序（位置递减）写入。
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void k2q_compact_lists_vf(
    const __gm__ uint8_t* state_matrix,  // [B, H, num_q_blocks, num_kv_blocks]
    int32_t kv_len, int32_t kv_block_size, int32_t num_q_blocks, int32_t num_kv_blocks, int32_t H, int32_t B,
    __gm__ int32_t* mask_block_cnt,  // [B, H, num_kv_blocks]
    __gm__ int32_t* full_block_cnt,  // [B, H, num_kv_blocks]
    __gm__ int32_t* block_id)        // [B, H, num_kv_blocks, num_q_blocks]
{
    const int32_t tid = AscendC::Simt::GetThreadIdx();
    const int32_t warp_id = tid / WARP_SIZE_CONST;
    const int32_t lane_id = tid % WARP_SIZE_CONST;

    const int32_t groups_per_bh = DIVUP(num_kv_blocks, WARP_SIZE_CONST);
    const int32_t total_groups = B * H * groups_per_bh;
    const int32_t grid_ceil = DIVUP(total_groups, WARP_LEVEL_NUM);
    for (int32_t pass = AscendC::Simt::GetBlockIdx(); pass < grid_ceil; pass += AscendC::Simt::GetBlockNum()) {
        const int32_t group = pass * WARP_LEVEL_NUM + warp_id;
        if (group >= total_groups) {
            continue;
        }
        const int32_t bh = group / groups_per_bh;
        const int32_t kv_group = group - bh * groups_per_bh;
        const int32_t kv_block = kv_group * WARP_SIZE_CONST + lane_id;
        if (kv_block >= num_kv_blocks) {
            continue;
        }

        // 矩阵列基址：state_matrix[bh, 0, 0, kv_block]，沿 q_block 维（步长 num_kv_blocks）扫描
        const __gm__ uint8_t* col_ptr = state_matrix + static_cast<int64_t>(bh) * num_q_blocks * num_kv_blocks +
                                        kv_block;
        const int32_t kv_start = kv_block * kv_block_size;
        const int32_t kv_end = min(kv_start + kv_block_size, kv_len);
        const bool is_kv_block_full = (kv_end - kv_start == kv_block_size);

        const int32_t cnt_offset = bh * num_kv_blocks + kv_block;
        const int32_t idx_base = cnt_offset * num_q_blocks;
        int32_t full_count = 0;
        int32_t mask_count = 0;
        for (int32_t q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
            const uint8_t q_block_state = col_ptr[static_cast<int64_t>(q_block_idx) * num_kv_blocks];
            if (q_block_state == STATE_FULL) {
                if (is_kv_block_full) {
                    block_id[idx_base + full_count] = q_block_idx;
                    full_count++;
                } else {
                    // 溢出 kv 边界的 kv_block 只能算 mask
                    block_id[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                    mask_count++;
                }
            } else if (q_block_state == STATE_MASK) {
                block_id[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                mask_count++;
            }
        }
        mask_block_cnt[cnt_offset] = mask_count;
        full_block_cnt[cnt_offset] = full_count;
    }
}

// =============================================================================
// extern "C" 入口：依据运行期 nFunc / qBlockSize / kvBlockSize 做模板分发
// =============================================================================
#define K2Q_LAUNCH(NF, QBS, KVBS, BS)                                                                      \
    asc_vf_call<create_k2q_block_sparse_from_func_kernel_vf<NF, QBS, KVBS>>(                               \
        dim3{static_cast<uint32_t>(BS)}, (__gm__ int32_t*)func_tensor, strideB, strideH, strideQ,         \
        strideF, qLen, kvLen, nFunc, qBlockSize, kvBlockSize, numQBlocks, numKvBlocks, kvColsPerFillTask,  \
        state_matrix, hNum, bNum)

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

// =============================================================================
// 小 shape 单阶段列式路径分发（仅依据 nFunc 模板化）
// =============================================================================
#define K2Q_SINGLE_LAUNCH(NF)                                                                             \
    asc_vf_call<k2q_single_phase_vf<NF>>(                                                                 \
        dim3{static_cast<uint32_t>(BLOCK_THREAD_NUM_256)}, (__gm__ int32_t*)func_tensor, strideB,         \
        strideH, strideQ, strideF, qLen, kvLen, nFunc, qBlockSize, kvBlockSize, numQBlocks, numKvBlocks,  \
        kvColsPerTask, hNum, bNum, (__gm__ int32_t*)mask_block_cnt, (__gm__ int32_t*)full_block_cnt,      \
        (__gm__ int32_t*)block_id)

#define DISPATCH_K2Q_SINGLE_BY_NFUNC() \
    switch (nFunc) {                   \
        case 2:                        \
            K2Q_SINGLE_LAUNCH(2);      \
            break;                     \
        case 4:                        \
            K2Q_SINGLE_LAUNCH(4);      \
            break;                     \
        case 6:                        \
            K2Q_SINGLE_LAUNCH(6);      \
            break;                     \
        case 8:                        \
            K2Q_SINGLE_LAUNCH(8);      \
            break;                     \
        case 10:                       \
            K2Q_SINGLE_LAUNCH(10);     \
            break;                     \
        case 12:                       \
            K2Q_SINGLE_LAUNCH(12);     \
            break;                     \
        case 14:                       \
            K2Q_SINGLE_LAUNCH(14);     \
            break;                     \
        case 16:                       \
            K2Q_SINGLE_LAUNCH(16);     \
            break;                     \
        case 18:                       \
            K2Q_SINGLE_LAUNCH(18);     \
            break;                     \
        case 20:                       \
            K2Q_SINGLE_LAUNCH(20);     \
            break;                     \
        case 22:                       \
            K2Q_SINGLE_LAUNCH(22);     \
            break;                     \
        case 24:                       \
            K2Q_SINGLE_LAUNCH(24);     \
            break;                     \
        case 26:                       \
            K2Q_SINGLE_LAUNCH(26);     \
            break;                     \
        case 28:                       \
            K2Q_SINGLE_LAUNCH(28);     \
            break;                     \
        case 30:                       \
            K2Q_SINGLE_LAUNCH(30);     \
            break;                     \
        case 32:                       \
            K2Q_SINGLE_LAUNCH(32);     \
            break;                     \
        default:                       \
            K2Q_SINGLE_LAUNCH(0);      \
            break;                     \
    }

extern "C" __global__ __aicore__ void create_k2q_block_sparse_from_func(GM_ADDR func_tensor,
                                                                        GM_ADDR mask_block_cnt, GM_ADDR full_block_cnt,
                                                                        GM_ADDR block_id, GM_ADDR workspace,
                                                                        GM_ADDR tiling)
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
    int32_t kvColsPerTask = tilingData.kvColsPerTask;
    int32_t kvColsPerFillTask = tilingData.kvColsPerFillTask;

    if (tilingData.useSinglePhase != 0) {
        // 小 shape 单阶段 warp-per-column 路径：直接有序输出，无状态矩阵 / SyncAll
        DISPATCH_K2Q_SINGLE_BY_NFUNC();
        return;
    }

    // 用户 workspace：状态矩阵 [B, H, numQBlocks, numKvBlocks]（uint8）
    __gm__ uint8_t* state_matrix = reinterpret_cast<__gm__ uint8_t*>(AscendC::GetUserWorkspace(workspace));

    // =========================================================================
    // 第一阶段：状态矩阵写入
    // =========================================================================
#ifdef ENABLE_WARP_LEVEL_OPT
    int32_t blockSize = WARP_LEVEL_NUM * WARP_SIZE_CONST;
    switch (qBlockSize) {
        case 32:
            DISPATCH_K2Q_BY_NFUNC(nFunc, 32, 0, blockSize);
            break;
        case 64:
            DISPATCH_K2Q_BY_NFUNC(nFunc, 64, 0, blockSize);
            break;
        case 128:
            DISPATCH_K2Q_BY_NFUNC(nFunc, 128, 0, blockSize);
            break;
        case 256:
            DISPATCH_K2Q_BY_NFUNC(nFunc, 256, 0, blockSize);
            break;
        default:
            blockSize = min(qBlockSize, static_cast<int32_t>(BLOCK_THREAD_NUM_256));
            DISPATCH_K2Q_BY_NFUNC(nFunc, 0, 0, blockSize);
            break;
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
        blockSize = min(qBlockSize, static_cast<int32_t>(BLOCK_THREAD_NUM_256));
        DISPATCH_K2Q_BY_NFUNC(nFunc, 0, 0, blockSize);
    }
#endif  // ENABLE_WARP_LEVEL_OPT

    // =========================================================================
    // 跨核硬同步：所有核完成状态矩阵写入后，再进入第二阶段
    // =========================================================================
    SyncAll();

    // =========================================================================
    // 第二阶段：有序紧缩输出
    // =========================================================================
    const int32_t compactBlockSize = WARP_LEVEL_NUM * WARP_SIZE_CONST;
    asc_vf_call<k2q_compact_lists_vf>(
        dim3{static_cast<uint32_t>(compactBlockSize)}, state_matrix, kvLen, kvBlockSize, numQBlocks, numKvBlocks, hNum,
        bNum, (__gm__ int32_t*)mask_block_cnt, (__gm__ int32_t*)full_block_cnt, (__gm__ int32_t*)block_id);
}
