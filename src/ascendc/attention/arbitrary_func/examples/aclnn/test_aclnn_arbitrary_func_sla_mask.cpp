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
 * @file test_aclnn_arbitrary_func_sla_mask.cpp
 * @brief 演示如何通过 aclnnBuildSlaMaskArbitraryFunc 接口调用 BuildSlaMaskArbitraryFunc 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_build_sla_mask_arbitrary_func.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 两个序列：[6 len, 2 tgt], [4 len, 1 tgt]，K1=2 / K2=3 / ctx=0，total_seqlen = 10
    //   seq0 的 history 边界 H=4；seq1 的 H=3
    const int64_t batchSize = 2;
    const int64_t slaK1 = 2;
    const int64_t slaK2 = 3;
    const int64_t contextualSeqLen = 0;
    const int64_t totalSeqLen = 10;

    std::vector<int32_t> offsetsHost = {0, 6, 10};
    std::vector<int32_t> numTargetsHost = {2, 1};
    std::vector<int32_t> outHost(totalSeqLen * 4, 0);

    std::vector<int64_t> offsetsShape = {3};
    std::vector<int64_t> numTargetsShape = {2};
    std::vector<int64_t> outShape = {1, 1, totalSeqLen, 4};
    void *offsetsDev = nullptr, *ntDev = nullptr, *outDev = nullptr;
    aclTensor *offsets = nullptr, *numTargets = nullptr, *out = nullptr;
    ret = CreateAclTensor(offsetsHost, offsetsShape, &offsetsDev, aclDataType::ACL_INT32, &offsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numTargetsHost, numTargetsShape, &ntDev, aclDataType::ACL_INT32, &numTargets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, outShape, &outDev, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize(offsets, numTargets, slaK1, slaK2, contextualSeqLen,
                                                         totalSeqLen, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildSlaMaskArbitraryFuncGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBuildSlaMaskArbitraryFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildSlaMaskArbitraryFunc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = totalSeqLen * 4 * sizeof(int32_t);
    ret = aclrtMemcpy(outHost.data(), outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("build_sla_mask_arbitrary_func output shape [1,1,%lld,4]\n", static_cast<long long>(totalSeqLen));
    // pos0（seq0 history 首行）：expect [0, 1, 3, 1]（Global[0,1) ∪ Local 空）
    LOG_PRINT("row pos0 : %d %d %d %d\n", outHost[0], outHost[1], outHost[2], outHost[3]);
    // pos4（seq0 第 1 个 target，H=4）：expect [0, 4, 4, 4]（只见 [0,4)，连自身不可见）
    LOG_PRINT("row pos4 : %d %d %d %d\n", outHost[16], outHost[17], outHost[18], outHost[19]);
    // pos9（seq1 唯一 target，H=3）：expect [6, 9, 9, 9]（只见 [6,9)）
    LOG_PRINT("row pos9 : %d %d %d %d\n", outHost[36], outHost[37], outHost[38], outHost[39]);

    aclDestroyTensor(offsets);
    aclDestroyTensor(numTargets);
    aclDestroyTensor(out);
    aclrtFree(offsetsDev);
    aclrtFree(ntDev);
    aclrtFree(outDev);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
