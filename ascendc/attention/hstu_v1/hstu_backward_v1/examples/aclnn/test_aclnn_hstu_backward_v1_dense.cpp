Copyright(c) Huawei Technologies Co., Ltd..txt
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
 * @file test_aclnn_hstu_dense_backward.cpp
 * @brief 演示如何通过 aclnnHstuDenseBackward 接口调用 HstuDenseBackward 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hstu_dense_backward.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

    // 参数: B=2, N=64, H=4, D=32, maskType=2(无mask)
    constexpr int64_t batchSize = 2;
    constexpr int64_t seqLen = 64;
    constexpr int64_t headNum = 4;
    constexpr int64_t headDim = 32;
    constexpr int64_t maskType = 2;
    constexpr int64_t maxSeqLen = seqLen;
    constexpr int64_t targetGroupSize = 0;
    constexpr double alpha = 1.0;
    // 注意：siluScale 不能传 0.0，算子会把 score 整体乘以该值，0.0 会导致输出全零。
    // 与 torch 插件一致，未显式指定时按 1.0/max_seq_len 计算。
    const double siluScaleReal = 1.0 / static_cast<double>(maxSeqLen);

    std::vector<int64_t> tensorShape = {batchSize, seqLen, headNum, headDim};
    int64_t numElements = GetShapeSize(tensorShape);

    // 生成输入数据
    std::vector<float> gradHost(numElements);
    std::vector<float> qHost(numElements);
    std::vector<float> kHost(numElements);
    std::vector<float> vHost(numElements);
    for (int64_t i = 0; i < numElements; ++i) {
        gradHost[i] = static_cast<float>(i % 7) / 7.0f;
        qHost[i] = static_cast<float>(i % 5) / 5.0f;
        kHost[i] = static_cast<float>(i % 3) / 3.0f;
        vHost[i] = static_cast<float>(i % 11) / 11.0f;
    }

    // 输出数据
    std::vector<float> qGradHost(numElements, 0);
    std::vector<float> kGradHost(numElements, 0);
    std::vector<float> vGradHost(numElements, 0);
    // attnBiasGrad 形状: [B, H, N, N]
    std::vector<int64_t> biasShape = {batchSize, headNum, seqLen, seqLen};
    int64_t biasNumElements = GetShapeSize(biasShape);
    std::vector<float> attnBiasGradHost(biasNumElements, 0);

    // 创建 Device 张量
    void *gradDeviceAddr = nullptr, *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr, *vDeviceAddr = nullptr;
    void *qGradDeviceAddr = nullptr, *kGradDeviceAddr = nullptr;
    void *vGradDeviceAddr = nullptr, *attnBiasGradDeviceAddr = nullptr;
    aclTensor *gradTensor = nullptr, *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *maskTensor = nullptr, *attnBiasTensor = nullptr;
    aclTensor *qGradTensor = nullptr, *kGradTensor = nullptr, *vGradTensor = nullptr, *attnBiasGradTensor = nullptr;

    ret = CreateAclTensor(gradHost, tensorShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &gradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qHost, tensorShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &qTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, tensorShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &kTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHost, tensorShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &vTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // mask 和 attnBias 为 None（maskType=2 时不需要 mask）
    maskTensor = nullptr;
    attnBiasTensor = nullptr;

    ret = CreateAclTensor(qGradHost, tensorShape, &qGradDeviceAddr, aclDataType::ACL_FLOAT, &qGradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGradHost, tensorShape, &kGradDeviceAddr, aclDataType::ACL_FLOAT, &kGradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vGradHost, tensorShape, &vGradDeviceAddr, aclDataType::ACL_FLOAT, &vGradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnBiasGradHost, biasShape, &attnBiasGradDeviceAddr, aclDataType::ACL_FLOAT,
                          &attnBiasGradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 获取 workspace 和执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    char layout[] = "normal";

    // 空张量占位（seqOffset / numContext / numTarget 在 dense 模式下不使用）
    aclTensor* emptyTensor = nullptr;

    ret = aclnnHstuDenseBackwardGetWorkspaceSize(
        gradTensor, qTensor, kTensor, vTensor, maskTensor, attnBiasTensor, emptyTensor, emptyTensor, emptyTensor,
        emptyTensor, layout, maskType, maxSeqLen, maxSeqLen, siluScaleReal, targetGroupSize, alpha, qGradTensor,
        kGradTensor, vGradTensor, attnBiasGradTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuDenseBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    // 回拷结果
    auto outSize = numElements * sizeof(float);
    ret = aclrtMemcpy(qGradHost.data(), outSize, qGradDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy qGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(kGradHost.data(), outSize, kGradDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy kGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(vGradHost.data(), outSize, vGradDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy vGrad failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_dense_backward output sample [0..3]: qGrad=%.4f %.4f kGrad=%.4f %.4f vGrad=%.4f %.4f\n",
              qGradHost[0], qGradHost[1], kGradHost[0], kGradHost[1], vGradHost[0], vGradHost[1]);

    // 清理资源
    aclDestroyTensor(gradTensor);
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(qGradTensor);
    aclDestroyTensor(kGradTensor);
    aclDestroyTensor(vGradTensor);
    aclDestroyTensor(attnBiasGradTensor);
    aclrtFree(gradDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(qGradDeviceAddr);
    aclrtFree(kGradDeviceAddr);
    aclrtFree(vGradDeviceAddr);
    aclrtFree(attnBiasGradDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
