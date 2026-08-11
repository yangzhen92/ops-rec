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
 * @file test_aclnn_hstu_forward.cpp
 * @brief 演示如何通过 aclnnHstuDenseForward 接口调用 HstuDenseForward 算子（normal 格式前向）
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_hstu_dense_forward.h"

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

    constexpr int64_t B = 1;
    constexpr int64_t S = 16;   // seq_len（max_seq_len 需覆盖 S）
    constexpr int64_t N = 2;    // head_num
    constexpr int64_t D = 32;   // head_dim（16 的倍数）
    constexpr int64_t MaxSeqLen = 16;
    constexpr int64_t MaskType = 0;  // 内置下三角 mask

    std::vector<int64_t> qkvShape = {B, S, N, D};
    auto qkvElemNum = GetShapeSize(qkvShape);
    std::vector<int16_t> qHostData(qkvElemNum);
    std::vector<int16_t> kHostData(qkvElemNum);
    std::vector<int16_t> vHostData(qkvElemNum);
    std::vector<int16_t> outHostData(qkvElemNum, 0);
    for (int64_t i = 0; i < qkvElemNum; ++i) {
        qHostData[i] = static_cast<int16_t>((i % 3 - 1) * 100);  // fp16 简单定值数据
        kHostData[i] = static_cast<int16_t>((i % 5 - 2) * 50);
        vHostData[i] = static_cast<int16_t>(i % 7);
    }

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qkvShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
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

    auto outSize = qkvElemNum * sizeof(int16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_dense_forward output sample [0,0,0,0..3]: %d %d %d %d\n", outHostData[0], outHostData[1],
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
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
