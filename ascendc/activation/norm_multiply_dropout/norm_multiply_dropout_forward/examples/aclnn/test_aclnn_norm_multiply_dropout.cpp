/**
 * @file test_aclnn_norm_multiply_dropout.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * aclnnNormMultiplyDropout 两段式接口调用示例。
 * dropout_ratio <= 1e-10 时不执行 dropout，mask 不消费，shape 传占位 {1}。
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_norm_multiply_dropout.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (cond) {           \
            expr;             \
        }                     \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
        printf("\n");               \
    } while (0)

int32_t deviceId = 0;
aclrtStream stream;

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, std::vector<T>& hostData,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = shape.empty() ? sizeof(T) : std::accumulate(shape.begin(), shape.end(), sizeof(T),
                                                            [](int64_t a, int64_t b) { return a * b; });
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // batch、col 及 tensor 初始化
    int64_t batch = 4;
    int64_t col = 512;
    std::vector<int64_t> xShape = {batch, col};
    std::vector<int64_t> weightShape = {col};
    std::vector<int64_t> meanShape = {batch};
    float eps = 1e-5f;
    float dropoutRatio = 0.0f;

    int InitRsp = Init(deviceId, &stream);
    CHECK_RET(InitRsp == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", InitRsp); return InitRsp);

    std::vector<float> xHostData(batch * col, 0.5f);
    std::vector<float> uHostData(batch * col, 0.5f);
    std::vector<float> weightHostData(col, 0.5f);
    std::vector<float> biasHostData(col, 0.5f);
    std::vector<uint8_t> maskHostData(1, 0);  // 不 dropout，mask 占位
    std::vector<float> outputHostData(batch * col, 0);
    std::vector<float> meanHostData(batch, 0);
    std::vector<float> varHostData(batch, 0);

    void *xDeviceAddr = nullptr, *uDeviceAddr = nullptr, *weightDeviceAddr = nullptr, *biasDeviceAddr = nullptr;
    void *maskDeviceAddr = nullptr, *outputDeviceAddr = nullptr, *meanDeviceAddr = nullptr, *varDeviceAddr = nullptr;
    aclTensor *x = nullptr, *u = nullptr, *weight = nullptr, *bias = nullptr, *mask = nullptr;
    aclTensor *output = nullptr, *mean = nullptr, *var = nullptr;

    int ret = 0;
    ret = CreateAclTensor(xShape, &xDeviceAddr, xHostData, ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xShape, &uDeviceAddr, uHostData, ACL_FLOAT, &u);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightShape, &weightDeviceAddr, weightHostData, ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightShape, &biasDeviceAddr, biasHostData, ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor({1}, &maskDeviceAddr, maskHostData, ACL_UINT8, &mask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xShape, &outputDeviceAddr, outputHostData, ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanShape, &meanDeviceAddr, meanHostData, ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanShape, &varDeviceAddr, varHostData, ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用第一段接口
    ret = aclnnNormMultiplyDropoutGetWorkspaceSize(x, u, weight, bias, mask, eps, dropoutRatio, output, mean, var,
                                                   &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormMultiplyDropoutGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnNormMultiplyDropout(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormMultiplyDropout failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("aclnnNormMultiplyDropout execute success\n");

    aclDestroyTensor(x);
    aclDestroyTensor(u);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(mask);
    aclDestroyTensor(output);
    aclDestroyTensor(mean);
    aclDestroyTensor(var);
    aclrtFree(xDeviceAddr);
    aclrtFree(uDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(maskDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
