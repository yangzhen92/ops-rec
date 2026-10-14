/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#pragma once

namespace Catlass::Kernel {

template <bool IS_ARBITRARY, bool IS_LOCAL, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET, bool ENABLE_STREAM_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, class QScheduler, class KScheduler, class Params>
CATLASS_DEVICE void InitializeForwardSchedulers(QScheduler& qScheduler, KScheduler& kScheduler,
                                                const Params& params, uint32_t maskHeads,
                                                uint32_t maxSeqLenQ, uint32_t maxSeqLenK,
                                                int32_t windowLeft, int32_t windowRight,
                                                uint32_t targetGroupSize, uint32_t blockM, uint32_t blockN)
{
    // Forward partitions the outer Q axis.  Once every batch has a single Q
    // block, batch/head ownership is computable in O(1); the inner K loop may
    // still contain multiple blocks without changing that ownership.
    qScheduler.EnableSingleOuterBlockFastPath(maxSeqLenQ != 0U && maxSeqLenQ <= blockM);
    if constexpr (IS_ARBITRARY) {
        if constexpr (ENABLE_STREAM_K) {
            qScheduler.EnablePackedSparseWorkload(params.ptrSparseInfo, maskHeads,
                                                  (maxSeqLenQ + blockM - 1U) / blockM);
        }
        kScheduler.EnablePackedSparse(params.ptrSparseInfo, maskHeads,
                                      (maxSeqLenQ + blockM - 1U) / blockM,
                                      (maxSeqLenK + blockN - 1U) / blockN);
    } else if constexpr (IS_LOCAL && !IS_CONTEXT && !IS_TARGET) {
        kScheduler.EnableLocal(windowLeft, windowRight, true);
    } else if constexpr (IS_CAUSAL) {
        if (maxSeqLenK >= maxSeqLenQ) {
            kScheduler.EnableCausal(params.ptrNumContext, params.ptrNumTarget, targetGroupSize, IS_CONTEXT, IS_TARGET);
        }
    }
    qScheduler.Init();
    kScheduler.Init(qScheduler);
}

template <bool IS_ARBITRARY, bool IS_LOCAL, bool IS_CAUSAL, bool IS_CONTEXT, bool IS_TARGET,
          uint32_t BLOCK_M, uint32_t BLOCK_N, class QScheduler, class KScheduler, class Params>
CATLASS_DEVICE void InitializeBackwardSchedulers(QScheduler& qScheduler, KScheduler& kScheduler,
                                                 const Params& params, uint32_t maskHeads,
                                                 uint32_t maxSeqLenQ, uint32_t maxSeqLenK,
                                                 int32_t windowLeft, int32_t windowRight,
                                                 uint32_t targetGroupSize, uint32_t blockM, uint32_t blockN)
{
    // Backward masked workload can vary along the inner Q axis, so retain the
    // stricter two-axis short-sequence condition from the scheduling policy.
    kScheduler.EnableSingleOuterBlockFastPath(maxSeqLenK != 0U && maxSeqLenQ != 0U &&
                                              maxSeqLenK <= blockN && maxSeqLenQ <= blockM);
    const bool shortQOrLongK = maxSeqLenQ != 0U &&
        (maxSeqLenQ <= blockM || maxSeqLenK >= maxSeqLenQ + blockM);
    kScheduler.EnableUniformOuterBlockFastPath(shortQOrLongK, maxSeqLenK);
    if constexpr (IS_TARGET && !IS_LOCAL) {
        kScheduler.EnableTargetWorkload(params.ptrNumContext, params.ptrNumTarget, targetGroupSize, IS_CONTEXT);
    } else if constexpr (IS_CAUSAL && !IS_LOCAL) {
        if (maxSeqLenK >= maxSeqLenQ) {
            kScheduler.EnableCausalWorkload();
        }
    }
    if constexpr (IS_ARBITRARY) {
        kScheduler.EnablePackedSparseWorkload(params.ptrSparseInfo, maskHeads,
                                              (maxSeqLenK + blockN - 1U) / blockN);
        qScheduler.EnablePackedSparse(params.ptrSparseInfo, maskHeads,
                                      (maxSeqLenK + blockN - 1U) / blockN,
                                      (maxSeqLenQ + blockM - 1U) / blockM);
    }
    // Do not prune backward local-window Q blocks in the scheduler. For a
    // sufficiently large positive delta-QK, an outer K block may have zero
    // intersecting Q blocks. The MMAD and epilogue loops must nevertheless
    // execute matching acquire/flush events for that K block; skipping the
    // entire inner loop deadlocks the mixed AIC/AIV pipeline. LocalMaskPredictor
    // still masks every out-of-window element, preserving zero contributions.
    kScheduler.Init();
}

}  // namespace Catlass::Kernel
