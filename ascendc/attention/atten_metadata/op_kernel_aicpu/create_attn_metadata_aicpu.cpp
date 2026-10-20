/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

/*!
 * \file create_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "create_attn_metadata.h"
#include "create_attn_metadata_aicpu.h"
#include "aicpu_common.h"
#include "fa_adjust_sinner_souter.h"
#include "attn_schedule_policy.h"

#define FA_KERNEL_STATUS_OK 0
#define FA_KERNEL_STATUS_PARAM_INVALID 1

using namespace optiling;

namespace aicpu {
uint32_t CreateAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    if (deterministic_ && packedMode_ && maskBlockCnt_ != nullptr && maskBlockCnt_->GetData() != nullptr &&
        fullBlockCnt_ != nullptr && fullBlockCnt_->GetData() != nullptr) {
        KERNEL_CHECK_FALSE(GenPackedMetadata(), FA_KERNEL_STATUS_PARAM_INVALID, "Generate packed metadata failed!");
        return FA_KERNEL_STATUS_OK;
    }

    if (maskMode_ == 0) {
        KERNEL_CHECK_FALSE(GenDenseMetadata(), FA_KERNEL_STATUS_PARAM_INVALID, "Generate dense HSTU metadata failed!");
        return FA_KERNEL_STATUS_OK;
    }

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");
    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    return FA_KERNEL_STATUS_OK;
}

bool CreateAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    const std::string opType = ctx.GetOpType();
    packedMode_ = opType == "CreateAttnMetadataPacked";
    forwardMode_ = opType == "CreateAttnMetadataForward";
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    if (packedMode_) {
        maskBlockCnt_ = ctx.Input(static_cast<uint32_t>(ParamId::maskBlockCnt));
        fullBlockCnt_ = ctx.Input(static_cast<uint32_t>(ParamId::fullBlockCnt));
    }
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) && GetAttrValue(ctx, "head_dim", headDim_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) && GetAttrValue(ctx, "l2_size", l2Size_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // HSTU 仅支持 MHA (G=1)，不支持 GQA 分核下沉。
    KERNEL_CHECK_FALSE(numHeadsQ_ == numHeadsKv_, false,
                       "HSTU mode requires MHA: num_heads_q must equal num_heads_kv (G=1)");

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    if (packedMode_) {
        GetAttrValueOpt(ctx, "packed_row_axis", packedRowAxis_);
        GetAttrValueOpt(ctx, "packed_mask_heads", packedMaskHeads_);
    }
    GetAttrValueOpt(ctx, "q_block_size", qBlockSize_);
    GetAttrValueOpt(ctx, "kv_block_size", kvBlockSize_);
    GetAttrValueOpt(ctx, "deterministic", deterministic_);
    return ParamsInit();
}

bool CreateAttnMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

void CreateAttnMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void CreateAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    // packed_row_axis identifies the scheduler's outer row axis. Forward
    // schedules Q rows (M); backward schedules K rows (N) and Q columns (M).
    if (qBlockSize_ > 0 && kvBlockSize_ > 0) {
        mBaseSize_ = packedRowAxis_ == 0 ? qBlockSize_ : kvBlockSize_;
        s2BaseSize_ = packedRowAxis_ == 0 ? kvBlockSize_ : qBlockSize_;
    } else if (forwardMode_ || (packedMode_ && packedRowAxis_ == 0)) {
        mBaseSize_ = headDim_ <= 128 ? 128U : 64U;
        s2BaseSize_ = headDim_ <= 128 ? 128U : 64U;
    } else {
        mBaseSize_ = headDim_ <= 128 ? 128U : 64U;
        s2BaseSize_ = headDim_ <= 128 ? 256U : 128U;
    }
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = static_cast<uint32_t>(ops_rec::attn_plan::schedule::L2BudgetBytes(l2Size_));
    param.fdOn = false;  // HSTU 不支持 FD / stream-K，FA 记录必落整行边界
    param.deterministic = deterministic_;
}
void CreateAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = batchSize_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.preToken = (winLeft_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winLeft_);
    baseInfo.nextToken = (winRight_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winRight_);
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    LoadActualQuerySeq();
    LoadActualKvSeq();
}

void CreateAttnMetadataCpuKernel::LoadActualQuerySeq()
{
    baseInfo.actualQuerySeqSize.clear();
    baseInfo.isCumulativeQuerySeq = (layoutQ_ == "TND" || layoutQ_ == "NTD");

    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(sequsedQ_, sequsedQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.querySeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeQuerySeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualQuerySeqSize.begin());
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1U;
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensQ_, cuSeqlensQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.querySeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeQuerySeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.querySeqSize = std::max(baseInfo.querySeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

void CreateAttnMetadataCpuKernel::LoadActualKvSeq()
{
    baseInfo.actualKvSeqSize.clear();
    baseInfo.isCumulativeKvSeq = (layoutKv_ == "TND" || layoutKv_ == "NTD");

    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(sequsedKv_, sequsedKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.kvSeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeKvSeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualKvSeqSize.begin());
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensKv_, cuSeqlensKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.kvSeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeKvSeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.kvSeqSize = std::max(baseInfo.kvSeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

bool CreateAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult& splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool CreateAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult& splitRes)
{
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum, static_cast<uint32_t>(aicCoreNum_),
                                  static_cast<uint32_t>(aivCoreNum_));
    const bool hasFdWork =
        std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](const load_balance::SectionStreamKFdResult& result) { return result.usedVecNum > 0U; });
    // HSTU sets fdOn=false. Avoid clearing the unused AIV metadata region; it
    // can be larger than the AIC region and is never read when HEAD_IS_FD=0.
    faMetadata.Clear(hasFdWork);

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    faMetadata.SetHeadMetadata(HEAD_AIV_CORE_NUM_INDEX, static_cast<uint32_t>(aivCoreNum_));
    faMetadata.SetHeadMetadata(HEAD_AIC_CORE_NUM_INDEX, static_cast<uint32_t>(aicCoreNum_));
    if (splitRes.streamK) {
        faMetadata.SetHeadMetadata(HEAD_DIRECT_MAGIC_INDEX, metadata::STREAM_K_MAGIC);
    }
    if (hasFdWork) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)};  // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        if (splitRes.streamK) {
            for (uint32_t aicIdx = 0; aicIdx < static_cast<uint32_t>(aicCoreNum_); ++aicIdx) {
                faMetadata.SetFaMetadata(secIdx, aicIdx, DIRECT_CORE_TASK_BEGIN_INDEX, faRes.taskBegin[aicIdx]);
                faMetadata.SetFaMetadata(secIdx, aicIdx, DIRECT_CORE_TASK_COUNT_INDEX, faRes.taskCount[aicIdx]);
            }
            continue;
        }
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bN2End[prevLastCore] : faRes.bN2End[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.gS1End[prevLastCore] : faRes.gS1End[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bN2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.gS1End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bN2Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.gS1Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
    return true;
}

bool CreateAttnMetadataCpuKernel::GenPackedMetadata()
{
    auto* output = static_cast<uint32_t*>(metadata_->GetData());
    const auto outputShape = metadata_->GetTensorShape();
    KERNEL_CHECK_FALSE(outputShape != nullptr && !outputShape->GetDimSizes().empty(), false,
                       "metadata output shape is invalid");
    const size_t outputWords = static_cast<size_t>(outputShape->GetDimSize(0));
    const uint32_t maskHeads = static_cast<uint32_t>(packedMaskHeads_);
    const uint32_t countElements = static_cast<uint32_t>(maskBlockCnt_->GetDataSize() / sizeof(int32_t));
    KERNEL_CHECK_FALSE(maskHeads > 0U && countElements % (baseInfo.batchSize * maskHeads) == 0U, false,
                       "packed count geometry is invalid");
    const uint32_t outerStride = countElements / (baseInfo.batchSize * maskHeads);
    const auto* mask = static_cast<const int32_t*>(maskBlockCnt_->GetData());
    const auto* full = static_cast<const int32_t*>(fullBlockCnt_->GetData());
    const auto countShape = maskBlockCnt_->GetTensorShape();
    KERNEL_CHECK_FALSE(countShape != nullptr, false, "packed count shape is null");
    uint64_t totalWeight = 0;
    uint32_t totalRows = 0;
    for (uint32_t b = 0; b < baseInfo.batchSize; ++b) {
        const auto& actualSeq = packedRowAxis_ == 0 ? baseInfo.actualQuerySeqSize : baseInfo.actualKvSeqSize;
        uint32_t seqBegin = 0U;
        uint32_t seqEnd = packedRowAxis_ == 0 ? baseInfo.querySeqSize : baseInfo.kvSeqSize;
        if (!actualSeq.empty()) {
            seqBegin = b == 0 ? 0U : actualSeq[b - 1];
            seqEnd = actualSeq[b];
        }
        const uint32_t seqLen = seqEnd - seqBegin;
        const uint32_t rows = (seqLen + mBaseSize_ - 1U) / mBaseSize_;
        for (uint32_t h = 0; h < static_cast<uint32_t>(numHeadsKv_); ++h) {
            const uint32_t mh = maskHeads == 1U ? 0U : h;
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t pos = (b * maskHeads + mh) * outerStride + row;
                totalWeight += static_cast<uint32_t>(std::max(0, mask[pos]) + std::max(0, full[pos]));
                ++totalRows;
            }
        }
    }

    const uint32_t scheduleCoreNum = static_cast<uint32_t>(aicCoreNum_);
    const size_t requiredWords = HEAD_METADATA_STRIDE + scheduleCoreNum * FA_METADATA_STRIDE;
    KERNEL_CHECK_FALSE(requiredWords <= outputWords, false, "metadata output is too small");
    std::fill(output, output + requiredWords, 0U);
    output[HEAD_SECTION_NUM_INDEX] = 1U;
    output[HEAD_M_BASE_SIZE_INDEX] = mBaseSize_;
    output[HEAD_S2_BASE_SIZE_INDEX] = s2BaseSize_;
    output[HEAD_AIV_CORE_NUM_INDEX] = static_cast<uint32_t>(aivCoreNum_);
    output[HEAD_AIC_CORE_NUM_INDEX] = scheduleCoreNum;
    output[HEAD_DIRECT_MAGIC_INDEX] = HSTU_DIRECT_MAGIC;

    for (uint32_t core = 0; core < scheduleCoreNum; ++core) {
        ops_rec::attn_plan::schedule::WeightedCoreRangeBuilder builder(
            totalRows, totalWeight, core, scheduleCoreNum, ops_rec::attn_plan::schedule::BoundaryPolicy::NEAREST);
        for (uint32_t batch = 0; batch < baseInfo.batchSize; ++batch) {
            const auto& actualSeq = packedRowAxis_ == 0 ? baseInfo.actualQuerySeqSize : baseInfo.actualKvSeqSize;
            uint32_t seqBegin = 0U;
            uint32_t seqEnd = packedRowAxis_ == 0 ? baseInfo.querySeqSize : baseInfo.kvSeqSize;
            if (!actualSeq.empty()) {
                seqBegin = batch == 0 ? 0U : actualSeq[batch - 1];
                seqEnd = actualSeq[batch];
            }
            const uint32_t rows = (seqEnd - seqBegin + mBaseSize_ - 1U) / mBaseSize_;
            for (uint32_t head = 0; head < static_cast<uint32_t>(numHeadsKv_); ++head) {
                const uint32_t mh = maskHeads == 1U ? 0U : head;
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t pos = (batch * maskHeads + mh) * outerStride + row;
                    const uint64_t rowWeight = static_cast<uint32_t>(std::max(0, mask[pos]) + std::max(0, full[pos]));
                    builder.Push(rowWeight);
                }
            }
        }
        const auto range = builder.Finish();
        const uint32_t record = HEAD_METADATA_STRIDE + core * FA_METADATA_STRIDE;
        output[record + DIRECT_CORE_TASK_BEGIN_INDEX] = range.beginBlock;
        output[record + DIRECT_CORE_TASK_COUNT_INDEX] = range.endBlock - range.beginBlock;
    }
    return true;
}

bool CreateAttnMetadataCpuKernel::GenDenseMetadata()
{
    load_balance::SectionStreamKResult splitRes{};
    bool success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, false, "Schedule dense metadata load balance failed!");
    return GenMetadata(splitRes);
}

namespace {
static const char* denseKernelType = "CreateAttnMetadata";
static const char* forwardKernelType = "CreateAttnMetadataForward";
static const char* packedKernelType = "CreateAttnMetadataPacked";
REGISTER_CPU_KERNEL(denseKernelType, CreateAttnMetadataCpuKernel);
REGISTER_CPU_KERNEL(forwardKernelType, CreateAttnMetadataCpuKernel);
REGISTER_CPU_KERNEL(packedKernelType, CreateAttnMetadataCpuKernel);
}  // namespace

}  // namespace aicpu
