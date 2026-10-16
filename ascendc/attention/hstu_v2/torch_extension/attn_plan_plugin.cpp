/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/interface/AclInterface.h>

#include "../../attn_plan/host/attn_plan.h"
#include "../common/hstu_tile_size.h"
#include "../common/hstu_schedule_policy.h"

using at::Tensor;
using ops_rec::attn_plan::DirectionPlanResult;

namespace {
ops_rec::attn_plan::TensorList MakeSparseMeta(const Tensor& af, const Tensor& seqOffsetQ, int64_t maxSeqLenQ,
                                              int64_t maxSeqLenK, int64_t qBlock, int64_t kvBlock, bool backward)
{
    const int64_t batch = seqOffsetQ.size(0) - 1;
    const int64_t heads = af.dim() == 2 ? 1 : af.size(1);
    const int64_t nq = (maxSeqLenQ + qBlock - 1) / qBlock;
    const int64_t nk = (maxSeqLenK + kvBlock - 1) / kvBlock;
    const int64_t outer = backward ? nk : nq;
    const int64_t inner = backward ? nq : nk;
    auto options = af.options().device(c10::kMeta).dtype(at::kInt);
    auto count = at::empty({batch, heads, outer}, options);
    auto placeholder = at::empty({1}, options);
    return {count,
            at::empty({batch, heads, outer, 2}, options),
            placeholder,
            at::empty_like(count),
            placeholder,
            placeholder,
            at::empty({batch, heads, outer, inner}, options),
            placeholder,
            at::empty({batch * heads * outer, inner, qBlock, kvBlock / 32}, options)};
}

DirectionPlanResult GetFwdAttnPlanMeta(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const c10::optional<Tensor>& af, const Tensor& seqOffsetQ, const Tensor&,
                                       int64_t maxSeqLenQ, int64_t maxSeqLenK, const c10::optional<Tensor>& numContext,
                                       const c10::optional<Tensor>& numTarget, int64_t, int64_t windowSizeLeft,
                                       int64_t windowSizeRight, int64_t maskType, bool deterministic)
{
    const auto tile = ops_rec::hstu::GetFwdTileSize(std::max(q.size(2), v.size(2)));
    const uint64_t innerBytes =
        static_cast<uint64_t>(k.size(0)) * k.size(1) * static_cast<uint64_t>(q.size(2) + v.size(2)) * q.element_size();
    const bool shortSequence = maxSeqLenQ <= static_cast<int64_t>(tile.m) && maxSeqLenK <= static_cast<int64_t>(tile.n);
    const uint64_t coreNum =
        static_cast<uint64_t>(c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE));
    const uint64_t qBlocks = (static_cast<uint64_t>(maxSeqLenQ) + tile.m - 1U) / tile.m;
    const uint64_t kBlocks = (static_cast<uint64_t>(maxSeqLenK) + tile.n - 1U) / tile.n;
    const uint64_t outerTasks = static_cast<uint64_t>(seqOffsetQ.size(0) - 1) * q.size(1) * qBlocks;
    const bool streamK = !deterministic && outerTasks < coreNum && kBlocks > 1U && outerTasks * kBlocks >= coreNum;
    const bool useAicpu =
        !shortSequence && (streamK || (deterministic && ::hstu::schedule::ShouldUseAicpuSchedule(
                                                            innerBytes, ops_rec::attn_plan::GetRuntimeL2Size())));
    auto sparse = af.has_value() && af->defined()
                      ? MakeSparseMeta(*af, seqOffsetQ, maxSeqLenQ, maxSeqLenK, tile.m, tile.n, false)
                      : ops_rec::attn_plan::TensorList{};
    auto metadata =
        useAicpu ? c10::optional<Tensor>(at::empty({1}, q.options().device(c10::kMeta).dtype(at::kInt))) : c10::nullopt;
    return {{tile.m, tile.n, tile.k}, std::move(sparse), std::move(metadata), useAicpu};
}

DirectionPlanResult GetBwdAttnPlanMeta(const Tensor& q, const Tensor&, const Tensor& v, const c10::optional<Tensor>& af,
                                       const Tensor& seqOffsetQ, const Tensor&, int64_t maxSeqLenQ, int64_t maxSeqLenK,
                                       const c10::optional<Tensor>& numContext, const c10::optional<Tensor>& numTarget,
                                       int64_t, int64_t windowSizeLeft, int64_t windowSizeRight, int64_t maskType,
                                       bool deterministic)
{
    const auto tile = ops_rec::hstu::GetBwdTileSize(std::max(q.size(2), v.size(2)));
    const uint64_t coreNum =
        static_cast<uint64_t>(c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE));
    const uint64_t qBlocks = (static_cast<uint64_t>(maxSeqLenQ) + tile.m - 1U) / tile.m;
    const uint64_t kBlocks = (static_cast<uint64_t>(maxSeqLenK) + tile.n - 1U) / tile.n;
    const uint64_t outerTasks = static_cast<uint64_t>(seqOffsetQ.size(0) - 1) * q.size(1) * kBlocks;
    const bool useAicpu = !deterministic && outerTasks < coreNum && qBlocks > 1U && outerTasks * qBlocks >= coreNum;
    auto sparse = af.has_value() && af->defined()
                      ? MakeSparseMeta(*af, seqOffsetQ, maxSeqLenQ, maxSeqLenK, tile.m, tile.n, true)
                      : ops_rec::attn_plan::TensorList{};
    auto metadata =
        useAicpu ? c10::optional<Tensor>(at::empty({1}, q.options().device(c10::kMeta).dtype(at::kInt))) : c10::nullopt;
    return {{tile.m, tile.n, tile.k}, std::move(sparse), std::move(metadata), useAicpu};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("get_fwd_attn_plan(Tensor q, Tensor k, Tensor v, Tensor? af, Tensor seq_offset_q, Tensor seq_offset_k, "
          "int max_seqlen_q, int max_seqlen_k, Tensor? num_context=None, Tensor? num_target=None, "
          "int target_group_size=1, int window_size_left=-1, int window_size_right=-1, int mask_type=2, "
          "bool deterministic=True) "
          "-> (int[], Tensor[], Tensor?, bool)");
    m.def("get_bwd_attn_plan(Tensor q, Tensor k, Tensor v, Tensor? af, Tensor seq_offset_q, Tensor seq_offset_k, "
          "int max_seqlen_q, int max_seqlen_k, Tensor? num_context=None, Tensor? num_target=None, "
          "int target_group_size=1, int window_size_left=-1, int window_size_right=-1, int mask_type=2, "
          "bool deterministic=True) "
          "-> (int[], Tensor[], Tensor?, bool)");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("get_fwd_attn_plan", &ops_rec::attn_plan::GetFwdAttnPlan);
    m.impl("get_bwd_attn_plan", &ops_rec::attn_plan::GetBwdAttnPlan);
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("get_fwd_attn_plan", &GetFwdAttnPlanMeta);
    m.impl("get_bwd_attn_plan", &GetBwdAttnPlanMeta);
}
