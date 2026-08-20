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
 * @file test_aclnn_hstu_forward_dense.cpp
 * @brief 演示如何通过 aclnnHstuDenseForward 接口调用 HSTU 前向融合注意力算子（normal/dense 数据格式）。
 *
 * 用例为两段式接口：
 *   GetWorkspaceSize -> aclrtMalloc(workspace) -> aclnnHstuDenseForward 执行 -> aclrtSynchronizeDevice
 * 数据均使用 fp32 简单定值数据（与 hstu_backward_v1 样例一致）。
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_hstu_dense_forward.h"
#include "test_aclnn_hstu_forward_common.h"

int TestHstuDense(aclrtStream stream)
{
    constexpr int64_t B = 1;
    constexpr int64_t S = 16;   // seq_len（max_seq_len 需覆盖 S）
    constexpr int64_t N = 2;    // head_num
    constexpr int64_t D = 32;   // head_dim（16 的倍数）
    constexpr int64_t MaxSeqLen = 16;
    constexpr int64_t MaskType = 0;  // 内置下三角 mask

    std::vector<int64_t> qkvShape = {B, S, N, D};
    auto qkvElemNum = GetShapeSize(qkvShape);
    std::vector<float> qHostData(qkvElemNum);
    std::vector<float> kHostData(qkvElemNum);
    std::vector<float> vHostData(qkvElemNum);
    std::vector<float> outHostData(qkvElemNum, 0);
    for (int64_t i = 0; i < qkvElemNum; ++i) {
        qHostData[i] = static_cast<float>(i % 3) / 3.0f;  // fp32 简单定值数据
        kHostData[i] = static_cast<float>(i % 5) / 5.0f;
        vHostData[i] = static_cast<float>(i % 7) / 7.0f;
    }

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    auto ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qkvShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHstuDenseForwardGetWorkspaceSize(q, k, v, nullptr, nullptr, MaskType, MaxSeqLen, 1.0 / MaxSeqLen,
                                                out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuDenseForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qkvElemNum * sizeof(float);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_dense_forward output sample [0,0,0,0..3]: %.4f %.4f %.4f %.4f\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_dense_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
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

    ret = TestHstuDense(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuDense failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
