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

#ifndef OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_H
#define OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_H

#include <cstdint>
#include <climits>

// =============================================================================
// 编译宏（与原始 RecSDK 生产构建保持一致）：
//   ENABLE_WARP_LEVEL_OPT : 启用 warp 级优化（每个 warp 独立处理一个 (b,h,block)）
//   DISABLE_REG_CACHE     : 关闭寄存器缓存（未定义 → 启用，将 func 值缓存到寄存器）
//   DISABLE_KV_RANGE_OPT  : 关闭 kv 范围遍历优化（未定义 → 启用，读取预计算的
//                           q_block_kv_min/max 跳过无交叠的 q_block）
// =============================================================================
#define ENABLE_WARP_LEVEL_OPT
// #define DISABLE_REG_CACHE
// #define DISABLE_KV_RANGE_OPT

// =============================================================================
// 共享常量（create_q2k / create_k2q / compute_q_block_kv_range 通用）
// =============================================================================
constexpr uint32_t BLOCK_THREAD_NUM_256 = 256;
constexpr int32_t WARP_LEVEL_NUM = 8;
constexpr int32_t WARP_SIZE_CONST = 32;
constexpr int32_t STATE_EMPTY = 0;
constexpr int32_t STATE_MASK = 1;
constexpr int32_t STATE_FULL = 2;

namespace common {
// (x + y - 1) / y
// 使用 constexpr 而非普通 inline：AscendC SIMT 设备侧编译对普通 inline 函数
// 可能不强制内联而生成外部符号引用（kernel 为独立链接单元），导致
// undefined symbol: common::DivUp(int, int)。constexpr 强制编译期内联，host/device 均可用。
constexpr int32_t DivUp(int32_t x, int32_t y)
{
    return (x + y - 1) / y;
}
}  // namespace common

#endif  // OPS_ASCEND_ATTENTION_BLOCK_SPARSE_FUNC_OP_KERNEL_COMMON_H
