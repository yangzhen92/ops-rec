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
============================================================================== */

/**
 * @file hstu_forward_v2.cpp
 * @brief HSTU Forward V2 算子 Kernel 实现
 * @description 实现算子在 AI Core 上的执行逻辑，包括:
 *              - Cube 核上的 MMAD 主循环 (QK、PV 计算)
 *              - Vector 核上的 Epilogue 主循环 (SiLU 激活、RAB 处理、输出转置)
 *              - 支持 HAS_RAB 和 BLOCK_K 两种模板参数分支
 */

#include "kernel_operator.h"
#include "catlass_hstu/kernel/fwd/forward_kernel_tiling.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_builder.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_mmad_mainloop.hpp"
#include "catlass_hstu/kernel/fwd/forward_kernel_epilogue_mainloop.hpp"
#include "catlass_hstu/kernel/mask/predictor_builder.hpp"

/**
 * @brief HSTU Forward V2 算子 Kernel 入口函数
 * @tparam HAS_RAB 是否有相对位置偏置
 * @tparam BLOCK_K K 方向的块大小 (128)
 * @param q Query 张量
 * @param k Key 张量
 * @param v Value 张量
 * @param mask 注意力掩码 (可选)
 * @param attnBias 注意力偏置 (RAB)
 * @param seq_offset_q Query 序列偏移量
 * @param seq_offset_k Key 序列偏移量
 * @param num_context 上下文长度 (可选)
 * @param num_target 目标长度 (可选)
 * @param metadata flash_attn_metadata 分核输出 (可选, 未传 → 旧设备现算分核)
 * @param attnOutput 输出: 注意力输出
 * @param workSpace 工作空间
 * @param tiling Tiling 数据
 * @description 算子的主入口函数，根据当前运行的核类型 (AIC/AIV) 选择执行:
 *              - AIC (Cube 核): 执行 MMAD 主循环
 *              - AIV (Vector 核): 执行 Epilogue 主循环
 */
template <bool HAS_RAB, bool IS_LOCAL, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY,
          bool ENABLE_STREAM_K, uint32_t BLOCK_K>
CATLASS_DEVICE void hstu_forward_v2_impl(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR attnBias,
                                         GM_ADDR seq_offset_q, GM_ADDR seq_offset_k, GM_ADDR num_context,
                                         GM_ADDR num_target, GM_ADDR metadata, GM_ADDR outputAccum, GM_ADDR sparseInfo,
                                         GM_ADDR attnOutput, GM_ADDR workSpace, GM_ADDR tiling)
{
    constexpr bool HAS_MASK = IS_LOCAL || IS_CAUSAL || IS_ARBITRARY;
    using namespace Catlass::Kernel;
    using namespace Catlass::Kernel::Mask;

    using Element = DTYPE_Q;
    using ElementOffset = DTYPE_SEQ_OFFSET_Q;
    using KernelConfig = ForwardKernelConfig<Arch::Ascend950, Element, ElementOffset, BLOCK_K, HAS_RAB, IS_LOCAL,
                                             IS_CAUSAL, IS_ARBITRARY>;
    constexpr bool USE_CAUSAL_SERPENTINE = !IS_LOCAL && !IS_CONTEXT && !IS_TARGET && (IS_CAUSAL || IS_ARBITRARY);
    using KernelBuilder = ForwardKernelBuilder<KernelConfig, USE_CAUSAL_SERPENTINE, ENABLE_STREAM_K>;

    using L1TileShape = typename KernelConfig::L1TileShape;
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    using Predictor = PredictorSelector<IS_LOCAL, IS_CAUSAL, IS_ARBITRARY, BLOCK_M, BLOCK_N, ElementOffset, true>;

    // metadata 驱动路径的调度器类型(与设备类型同参: 行轴块=BLOCK_M=Cq, 列轴块=BLOCK_N=Rk)
    using MetaQBlockScheduler = Catlass::Gemm::Block::MetadataOuterBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>;
    using MetaKBlockScheduler = Catlass::Gemm::Block::InnerBlockScheduler<MetaQBlockScheduler, BLOCK_M, BLOCK_N, true>;

    if ASCEND_IS_AIC {
        using QBlockScheduler = typename KernelBuilder::QBlockScheduler;
        using KBlockScheduler = typename KernelBuilder::KBlockScheduler;

        using BlockMmadQK = typename KernelBuilder::BlockMmadQK;
        using BlockMmadPV = typename KernelBuilder::BlockMmadPV;

        if constexpr (ENABLE_STREAM_K) {
            using MmadMainLoopKernel =
                ForwardMmadMainloop<BlockMmadQK, BlockMmadPV, QBlockScheduler, KBlockScheduler, Predictor, IS_LOCAL,
                                    IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, true>;
            typename MmadMainLoopKernel::Params params{q,           k,          v,           seq_offset_q, seq_offset_k,
                                                       num_context, num_target, outputAccum, metadata,     sparseInfo};
            MmadMainLoopKernel kernel(tiling);
            kernel(params);
        } else if (metadata != nullptr) {
            using MmadMainLoopKernel =
                ForwardMmadMainloop<BlockMmadQK, BlockMmadPV, MetaQBlockScheduler, MetaKBlockScheduler, Predictor,
                                    IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, ENABLE_STREAM_K>;
            typename MmadMainLoopKernel::Params params{q,           k,          v,       seq_offset_q, seq_offset_k,
                                                       num_context, num_target, nullptr, metadata,     sparseInfo};
            MmadMainLoopKernel kernel(tiling);
            kernel(params);
        } else {
            using MmadMainLoopKernel =
                ForwardMmadMainloop<BlockMmadQK, BlockMmadPV, QBlockScheduler, KBlockScheduler, Predictor, IS_LOCAL,
                                    IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, ENABLE_STREAM_K>;
            typename MmadMainLoopKernel::Params params{q,           k,          v,       seq_offset_q, seq_offset_k,
                                                       num_context, num_target, nullptr, nullptr,      sparseInfo};
            MmadMainLoopKernel kernel(tiling);
            kernel(params);
        }
    } else {
        using QBlockScheduler = typename KernelBuilder::QBlockScheduler;
        using KBlockScheduler = typename KernelBuilder::KBlockScheduler;

        using BlockEpilogueQK = typename KernelBuilder::BlockEpilogueQK;
        using BlockEpiloguePV = typename KernelBuilder::BlockEpiloguePV;
        using StreamKReduceEpilogue = typename KernelBuilder::StreamKReduceEpilogue;

        if constexpr (ENABLE_STREAM_K) {
            using EpilogueMainLoopKernel =
                ForwardEpilogueMainloop<BlockEpilogueQK, BlockEpiloguePV, StreamKReduceEpilogue, QBlockScheduler,
                                        KBlockScheduler, Predictor, L1TileShape, IS_LOCAL, IS_CONTEXT, IS_TARGET,
                                        IS_ARBITRARY, true>;
            typename EpilogueMainLoopKernel::Params params{attnBias,    seq_offset_q, seq_offset_k,
                                                           num_context, num_target,   attnOutput,
                                                           outputAccum, metadata,     sparseInfo};
            EpilogueMainLoopKernel kernel(tiling);
            kernel(params);
        } else if (metadata != nullptr) {
            using EpilogueMainLoopKernel =
                ForwardEpilogueMainloop<BlockEpilogueQK, BlockEpiloguePV, StreamKReduceEpilogue, MetaQBlockScheduler,
                                        MetaKBlockScheduler, Predictor, L1TileShape, IS_LOCAL, IS_CONTEXT, IS_TARGET,
                                        IS_ARBITRARY, ENABLE_STREAM_K>;
            typename EpilogueMainLoopKernel::Params params{attnBias,    seq_offset_q, seq_offset_k,
                                                           num_context, num_target,   attnOutput,
                                                           outputAccum, metadata,     sparseInfo};
            EpilogueMainLoopKernel kernel(tiling);
            kernel(params);
        } else {
            using EpilogueMainLoopKernel =
                ForwardEpilogueMainloop<BlockEpilogueQK, BlockEpiloguePV, StreamKReduceEpilogue, QBlockScheduler,
                                        KBlockScheduler, Predictor, L1TileShape, IS_LOCAL, IS_CONTEXT, IS_TARGET,
                                        IS_ARBITRARY, ENABLE_STREAM_K>;
            typename EpilogueMainLoopKernel::Params params{attnBias,    seq_offset_q, seq_offset_k,
                                                           num_context, num_target,   attnOutput,
                                                           outputAccum, nullptr,      sparseInfo};
            EpilogueMainLoopKernel kernel(tiling);
            kernel(params);
        }
    }
}

template <bool HAS_RAB, bool IS_LOCAL, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY,
          bool ENABLE_STREAM_K, uint32_t BLOCK_K>
CATLASS_GLOBAL void hstu_forward_v2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR attnBias,
                                    GM_ADDR seq_offset_q, GM_ADDR seq_offset_k, GM_ADDR num_context, GM_ADDR num_target,
                                    GM_ADDR metadata, GM_ADDR outputAccum, GM_ADDR sparseInfo, GM_ADDR attnOutput,
                                    GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(HSTU_V2_TASK_TYPE);
    hstu_forward_v2_impl<HAS_RAB, IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, ENABLE_STREAM_K, BLOCK_K>(
        q, k, v, mask, attnBias, seq_offset_q, seq_offset_k, num_context, num_target, metadata, outputAccum, sparseInfo,
        attnOutput, workSpace, tiling);
}
