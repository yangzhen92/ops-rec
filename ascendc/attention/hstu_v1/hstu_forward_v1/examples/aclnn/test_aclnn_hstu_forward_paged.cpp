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

/**
 * @file test_aclnn_hstu_forward_paged.cpp
 * @brief 演示如何通过 aclnnHstuPagedForward 接口调用 HSTU 前向融合注意力算子（paged 数据格式）。
 *
 * 用例为两段式接口：
 *   GetWorkspaceSize -> aclrtMalloc(workspace) -> aclnnHstuPagedForward 执行 -> aclrtSynchronizeDevice
 * 数据均使用 fp32 简单定值数据（与 hstu_backward_v1 样例一致）。
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_hstu_paged_forward.h"
#include "test_aclnn_hstu_forward_common.h"

int TestHstuPaged(aclrtStream stream)
{
    // paged 数据构造（与 pytest test_hstu_paged_graph 一致，已实机验证）：
    // 每个 batch：历史 token 数（已存入 kv_cache）new_hist_lens = [32, 48]，
    // 目标（新生成）token 数 num_targets = [4, 6]；
    // page_size 必须 ∈ {32, 64, 128, 256}（tiling 层 BLOCK_HEIGHT=256 / MIN_PAGE_SIZE=32 约束）
    constexpr int64_t N = 4;              // head_num
    constexpr int64_t D = 32;             // head_dim（16 的倍数）
    constexpr int64_t PageSize = 32;      // 每页 KV 容量
    constexpr int64_t Batch = 2;
    constexpr int64_t Total = 90;         // (32+4) + (48+6)
    constexpr int64_t MaxSeqLen = 54;     // max(36, 54)
    constexpr int64_t NumPages = 3;       // ceil(32/32) + ceil(48/32) = 1 + 2
    constexpr int64_t MaskType = 0;       // 内置下三角 mask
    constexpr int64_t TargetGroupSize = 1;  // num_target 定义时必须 > 0（torch 扩展层同约束）

    std::vector<int64_t> qkvShape = {Total, N, D};
    std::vector<int64_t> kvCacheShape = {NumPages, 2, PageSize, N, D};
    std::vector<int64_t> offsetShape = {Batch + 1};    // seq_offset_q/k：{0, 36, 90}
    std::vector<int64_t> offsetTShape = {Batch + 1};   // seq_offset_t：{0, 4, 10}
    std::vector<int64_t> numTargetShape = {Batch};     // num_context/num_target：[2]

    auto qkvElemNum = GetShapeSize(qkvShape);
    auto kvCacheElemNum = GetShapeSize(kvCacheShape);
    std::vector<float> qHostData(qkvElemNum);
    std::vector<float> kHostData(qkvElemNum);
    std::vector<float> vHostData(qkvElemNum);
    std::vector<float> kvCacheHostData(kvCacheElemNum);
    std::vector<float> outHostData(qkvElemNum, 0);
    for (int64_t i = 0; i < qkvElemNum; ++i) {
        qHostData[i] = static_cast<float>(i % 3) / 3.0f;  // fp32 简单定值数据
        kHostData[i] = static_cast<float>(i % 5) / 5.0f;
        vHostData[i] = static_cast<float>(i % 7) / 7.0f;
    }
    for (int64_t i = 0; i < kvCacheElemNum; ++i) {
        kvCacheHostData[i] = static_cast<float>(i % 11) / 11.0f;
    }

    std::vector<int32_t> seqOffsetQHostData = {0, 36, 90};
    std::vector<int32_t> seqOffsetKHostData = {0, 36, 90};
    std::vector<int32_t> seqOffsetTHostData = {0, 4, 10};
    std::vector<int32_t> pageOffsetsHostData = {0, 1, 3};   // 每个 batch 在 page_ids 中的起始页偏移
    std::vector<int32_t> pageIdsHostData = {0, 1, 2};       // 全局页号
    std::vector<int32_t> lastPageLenHostData = {32, 16};    // 每个 batch 末页有效长度（不足 page_size 时；恰好满页取 page_size）
    std::vector<int32_t> numContextHostData = {0, 0};
    std::vector<int32_t> numTargetHostData = {4, 6};

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr, *outDeviceAddr = nullptr;
    void *kvCacheDeviceAddr = nullptr;
    void *seqOffsetQDeviceAddr = nullptr, *seqOffsetKDeviceAddr = nullptr, *seqOffsetTDeviceAddr = nullptr;
    void *pageOffsetsDeviceAddr = nullptr, *pageIdsDeviceAddr = nullptr, *lastPageLenDeviceAddr = nullptr;
    void *numContextDeviceAddr = nullptr, *numTargetDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    aclTensor *kvCache = nullptr;
    aclTensor *seqOffsetQ = nullptr, *seqOffsetK = nullptr, *seqOffsetT = nullptr;
    aclTensor *pageOffsets = nullptr, *pageIds = nullptr, *lastPageLen = nullptr;
    aclTensor *numContext = nullptr, *numTarget = nullptr;
    auto ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qkvShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvCacheHostData, kvCacheShape, &kvCacheDeviceAddr, aclDataType::ACL_FLOAT, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetQHostData, offsetShape, &seqOffsetQDeviceAddr, aclDataType::ACL_INT32, &seqOffsetQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetKHostData, offsetShape, &seqOffsetKDeviceAddr, aclDataType::ACL_INT32, &seqOffsetK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetTHostData, offsetTShape, &seqOffsetTDeviceAddr, aclDataType::ACL_INT32, &seqOffsetT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pageOffsetsHostData, offsetShape, &pageOffsetsDeviceAddr, aclDataType::ACL_INT32,
                          &pageOffsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pageIdsHostData, offsetShape, &pageIdsDeviceAddr, aclDataType::ACL_INT32, &pageIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lastPageLenHostData, numTargetShape, &lastPageLenDeviceAddr, aclDataType::ACL_INT32,
                          &lastPageLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numContextHostData, numTargetShape, &numContextDeviceAddr, aclDataType::ACL_INT32,
                          &numContext);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numTargetHostData, numTargetShape, &numTargetDeviceAddr, aclDataType::ACL_INT32, &numTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHstuPagedForwardGetWorkspaceSize(
        q, k, v, nullptr /*mask*/, nullptr /*attn_bias*/,
        seqOffsetQ, seqOffsetK, seqOffsetT,
        kvCache, pageOffsets, pageIds, lastPageLen,
        numContext, numTarget,
        MaskType, MaxSeqLen, MaxSeqLen, 1.0 / MaxSeqLen,
        TargetGroupSize, 1.0 /*alpha*/, false /*deterministic*/,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuPagedForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuPagedForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuPagedForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qkvElemNum * sizeof(float);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_paged_forward output sample [0,0,0,0..3]: %.4f %.4f %.4f %.4f\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_paged_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(seqOffsetQ);
    aclDestroyTensor(seqOffsetK);
    aclDestroyTensor(seqOffsetT);
    aclDestroyTensor(pageOffsets);
    aclDestroyTensor(pageIds);
    aclDestroyTensor(lastPageLen);
    aclDestroyTensor(numContext);
    aclDestroyTensor(numTarget);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(seqOffsetQDeviceAddr);
    aclrtFree(seqOffsetKDeviceAddr);
    aclrtFree(seqOffsetTDeviceAddr);
    aclrtFree(pageOffsetsDeviceAddr);
    aclrtFree(pageIdsDeviceAddr);
    aclrtFree(lastPageLenDeviceAddr);
    aclrtFree(numContextDeviceAddr);
    aclrtFree(numTargetDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = TestHstuPaged(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuPaged failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
