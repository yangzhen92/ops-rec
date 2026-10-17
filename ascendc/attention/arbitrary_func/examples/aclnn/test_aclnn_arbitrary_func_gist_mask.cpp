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
 * @file test_aclnn_arbitrary_func_gist_mask.cpp
 * @brief 演示如何通过 aclnnBuildGistMaskArbitraryFunc 接口调用 BuildGistMaskArbitraryFunc 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_build_gist_mask_arbitrary_func.h"

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

    const int64_t totalSeqLen = 10;

    std::vector<int32_t> offsetsHost = {0, 10};
    std::vector<int32_t> preDocLensHost = {3};
    std::vector<int32_t> gistStartsHost = {5};
    std::vector<int32_t> gistEndsHost = {7};
    std::vector<int32_t> outHost(totalSeqLen * 4, 0);

    std::vector<int64_t> offsetsShape = {2};
    std::vector<int64_t> cfgShape = {1};
    std::vector<int64_t> outShape = {1, 1, totalSeqLen, 4};
    void *offsetsDev = nullptr, *pdlDev = nullptr, *gsDev = nullptr, *geDev = nullptr, *outDev = nullptr;
    aclTensor *offsets = nullptr, *preDocLens = nullptr, *gistStarts = nullptr, *gistEnds = nullptr, *out = nullptr;
    ret = CreateAclTensor(offsetsHost, offsetsShape, &offsetsDev, aclDataType::ACL_INT32, &offsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(preDocLensHost, cfgShape, &pdlDev, aclDataType::ACL_INT32, &preDocLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gistStartsHost, cfgShape, &gsDev, aclDataType::ACL_INT32, &gistStarts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gistEndsHost, cfgShape, &geDev, aclDataType::ACL_INT32, &gistEnds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, outShape, &outDev, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize(offsets, preDocLens, gistStarts, gistEnds, totalSeqLen, out,
                                                          &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildGistMaskArbitraryFuncGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBuildGistMaskArbitraryFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBuildGistMaskArbitraryFunc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = totalSeqLen * 4 * sizeof(int32_t);
    ret = aclrtMemcpy(outHost.data(), outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("build_gist_mask_arbitrary_func output shape [1,1,%lld,4]\n", static_cast<long long>(totalSeqLen));
    // pos0 (pre_doc): expect [0, 1, 0, 0]
    LOG_PRINT("row pos0 : %d %d %d %d\n", outHost[0], outHost[1], outHost[2], outHost[3]);
    // pos4 (bridge): expect [0, 3, 3, 5]
    LOG_PRINT("row pos4 : %d %d %d %d\n", outHost[16], outHost[17], outHost[18], outHost[19]);
    // pos6 (gist): expect [0, 7, 0, 0]
    LOG_PRINT("row pos6 : %d %d %d %d\n", outHost[24], outHost[25], outHost[26], outHost[27]);
    // pos9 (post-gist): expect [0, 3, 5, 7]
    LOG_PRINT("row pos9 : %d %d %d %d\n", outHost[36], outHost[37], outHost[38], outHost[39]);

    aclDestroyTensor(offsets);
    aclDestroyTensor(preDocLens);
    aclDestroyTensor(gistStarts);
    aclDestroyTensor(gistEnds);
    aclDestroyTensor(out);
    aclrtFree(offsetsDev);
    aclrtFree(pdlDev);
    aclrtFree(gsDev);
    aclrtFree(geDev);
    aclrtFree(outDev);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
