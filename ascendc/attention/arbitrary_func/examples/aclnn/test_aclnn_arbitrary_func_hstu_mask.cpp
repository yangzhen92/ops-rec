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
 * @file test_aclnn_arbitrary_func_hstu_mask.cpp
 * @brief 演示如何通过 aclnnBuildHstuMaskArbitraryFunc 接口调用 BuildHstuMaskArbitraryFunc 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_build_hstu_mask_arbitrary_func.h"

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

    // 两个序列：[2 ctx, 1 tgt], [3 ctx, 2 tgt]，无 history，total_seqlen = 8
    const int64_t batchSize = 2;
    const int64_t totalSeqLen = 8;
    const int64_t targetGroupSize = 2;

    std::vector<int32_t> offsetsHost = {0, 3, 8};
    std::vector<int32_t> ctxLensHost = {2, 3};
    std::vector<int32_t> tgtLensHost = {1, 2};
    std::vector<int32_t> outHost(totalSeqLen * 4, 0);

    std::vector<int64_t> offsetsShape = {3};
    std::vector<int64_t> lensShape = {2};
    std::vector<int64_t> outShape = {1, 1, totalSeqLen, 4};
    void *offsetsDev = nullptr, *ctxDev = nullptr, *tgtDev = nullptr, *outDev = nullptr;
    aclTensor *offsets = nullptr, *ctxLens = nullptr, *tgtLens = nullptr, *out = nullptr;
    ret = CreateAclTensor(offsetsHost, offsetsShape, &offsetsDev, aclDataType::ACL_INT32, &offsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ctxLensHost, lensShape, &ctxDev, aclDataType::ACL_INT32, &ctxLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tgtLensHost, lensShape, &tgtDev, aclDataType::ACL_INT32, &tgtLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, outShape, &outDev, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnBuildHstuMaskArbitraryFuncGetWorkspaceSize(offsets, ctxLens, tgtLens, targetGroupSize, totalSeqLen, out,
                                                  &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildHstuMaskArbitraryFuncGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBuildHstuMaskArbitraryFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildHstuMaskArbitraryFunc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = totalSeqLen * 4 * sizeof(int32_t);
    ret = aclrtMemcpy(outHost.data(), outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("build_hstu_mask_arbitrary_func output shape [1,1,%lld,4]\n", static_cast<long long>(totalSeqLen));
    LOG_PRINT("first row (pos0 ctx): %d %d %d %d\n", outHost[0], outHost[1], outHost[2], outHost[3]);
    LOG_PRINT("last  row (pos7 tgt): %d %d %d %d\n", outHost[28], outHost[29], outHost[30], outHost[31]);

    aclDestroyTensor(offsets);
    aclDestroyTensor(ctxLens);
    aclDestroyTensor(tgtLens);
    aclDestroyTensor(out);
    aclrtFree(offsetsDev);
    aclrtFree(ctxDev);
    aclrtFree(tgtDev);
    aclrtFree(outDev);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
