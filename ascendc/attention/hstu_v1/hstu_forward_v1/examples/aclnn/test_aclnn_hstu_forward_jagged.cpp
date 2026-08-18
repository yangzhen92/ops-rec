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
 * @file test_aclnn_hstu_forward_jagged.cpp
 * @brief 演示如何通过 aclnnHstuJaggedForward 接口调用 HSTU 前向融合注意力算子（jagged 数据格式，delta 语义）。
 *
 * 用例为两段式接口：
 *   GetWorkspaceSize -> aclrtMalloc(workspace) -> aclnnHstuJaggedForward 执行 -> aclrtSynchronizeDevice
 * 数据均使用 int16_t 模拟 fp16 的简单定值数据。
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_hstu_jagged_forward.h"
#include "test_aclnn_hstu_forward_common.h"

int TestHstuJagged(aclrtStream stream)
{
    // delta 数据构造：q 段长 [4, 6]，k/v 段长 [6, 8]（与 pytest test_hstu_jagged_delta_graph 一致）
    constexpr int64_t Batch = 2;        // 序列段数（seq_offset 段数，num_context/num_target 长度）
    constexpr int64_t N = 4;            // head_num
    constexpr int64_t D = 32;           // head_dim（16 的倍数）
    constexpr int64_t TotalQ = 10;      // sum([4, 6])
    constexpr int64_t TotalK = 14;      // sum([6, 8])
    constexpr int64_t MaxSeqLenQ = 6;   // max(q 段长)
    constexpr int64_t MaxSeqLenK = 8;   // max(k 段长)
    constexpr int64_t MaskType = 0;     // 内置下三角 mask

    std::vector<int64_t> qShape = {TotalQ, N, D};
    std::vector<int64_t> kvShape = {TotalK, N, D};
    auto qElemNum = GetShapeSize(qShape);
    auto kvElemNum = GetShapeSize(kvShape);
    std::vector<int16_t> qHostData(qElemNum);
    std::vector<int16_t> kHostData(kvElemNum);
    std::vector<int16_t> vHostData(kvElemNum);
    std::vector<int16_t> outHostData(qElemNum, 0);
    for (int64_t i = 0; i < qElemNum; ++i) {
        qHostData[i] = static_cast<int16_t>((i % 3 - 1) * 100);  // fp16 简单定值数据
    }
    for (int64_t i = 0; i < kvElemNum; ++i) {
        kHostData[i] = static_cast<int16_t>((i % 5 - 2) * 50);
        vHostData[i] = static_cast<int16_t>(i % 7);
    }

    std::vector<int64_t> offsetShape = {Batch + 1};
    std::vector<int64_t> numCtxShape = {Batch};
    std::vector<int32_t> seqOffsetQHostData = {0, 4, 10};
    std::vector<int32_t> seqOffsetKHostData = {0, 6, 14};
    // num_context/num_target 按 kernel 契约必须传入有效张量（不能为 nullptr）：
    // atlas950 jagged kernel 对这两个 GM 指针无条件解引用（PreInit/FillTaskInfo），
    // 本用例不启用 context/target 掩码，传全零张量（与 torch 适配层无值时的行为一致）
    std::vector<int32_t> numContextHostData(Batch, 0);
    std::vector<int32_t> numTargetHostData(Batch, 0);

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr, *outDeviceAddr = nullptr;
    void *seqOffsetQDeviceAddr = nullptr, *seqOffsetKDeviceAddr = nullptr;
    void *numContextDeviceAddr = nullptr, *numTargetDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    aclTensor *seqOffsetQ = nullptr, *seqOffsetK = nullptr;
    aclTensor *numContext = nullptr, *numTarget = nullptr;
    auto ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetQHostData, offsetShape, &seqOffsetQDeviceAddr, aclDataType::ACL_INT32, &seqOffsetQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetKHostData, offsetShape, &seqOffsetKDeviceAddr, aclDataType::ACL_INT32, &seqOffsetK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numContextHostData, numCtxShape, &numContextDeviceAddr, aclDataType::ACL_INT32, &numContext);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numTargetHostData, numCtxShape, &numTargetDeviceAddr, aclDataType::ACL_INT32, &numTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // mask / attn_bias 为可选入参，不使用传 nullptr（kernel 编译期裁剪，不读 GM）；
    // num_context / num_target 必须传有效张量（kernel 无条件解引用，见上注释），本用例传全零
    ret = aclnnHstuJaggedForwardGetWorkspaceSize(
        q, k, v, nullptr /*mask*/, nullptr /*attn_bias*/,
        seqOffsetQ, seqOffsetK, numContext, numTarget,
        MaskType, MaxSeqLenQ, MaxSeqLenK, 1.0 / MaxSeqLenQ,
        0 /*target_group_size*/, 1.0 /*alpha*/, false /*deterministic*/,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuJaggedForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuJaggedForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuJaggedForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qElemNum * sizeof(int16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_jagged_forward output sample [0,0,0,0..3]: %d %d %d %d\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_jagged_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclDestroyTensor(seqOffsetQ);
    aclDestroyTensor(seqOffsetK);
    aclDestroyTensor(numContext);
    aclDestroyTensor(numTarget);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(seqOffsetQDeviceAddr);
    aclrtFree(seqOffsetKDeviceAddr);
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

    ret = TestHstuJagged(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuJagged failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
