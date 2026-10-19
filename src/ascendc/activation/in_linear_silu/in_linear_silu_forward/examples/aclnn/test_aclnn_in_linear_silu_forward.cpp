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
 * @file test_aclnn_in_linear_silu.cpp
 * @brief 演示如何通过 aclnnInLinearSilu 接口调用 InLinearSilu 算子
 *
 * 功能：计算 mixed = x @ weight^T + bias，对 mixed 做 Silu 激活后，按
 * split_arg_list 拆分为 user、value、query、key，并额外输出 linear_output
 * （MatMul 的中间结果，供反向计算使用）。
 *
 * 编译/运行前请确保已部署 CANN 环境，并链接包含 aclnn_in_linear_silu.h 的
 * 头文件路径。
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_in_linear_silu.h"

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

    // 满足算子约束：k 为 16 的整数倍，n 为 4*k 的整数倍，split 成员为 16 的整数倍且总和等于 n
    constexpr int64_t M = 4;
    constexpr int64_t K = 16;
    constexpr int64_t N = 64;
    const std::vector<int64_t> splitArgList = {16, 16, 16, 16};

    // x / weight / 输出均为 FLOAT16，bias 为 FLOAT32
    std::vector<aclFloat16> xHostData(M * K);
    std::vector<aclFloat16> weightHostData(N * K);
    std::vector<float> biasHostData(N);
    for (int64_t i = 0; i < M * K; ++i) {
        xHostData[i] = aclFloatToFloat16(static_cast<float>(i % 8 - 4) / 2.0f);
    }
    for (int64_t i = 0; i < N * K; ++i) {
        weightHostData[i] = aclFloatToFloat16(static_cast<float>(i % 16 - 8) / 4.0f);
    }
    for (int64_t i = 0; i < N; ++i) {
        biasHostData[i] = 0.5f;
    }

    // 输出 host 缓冲（初始化为 0，算子执行后回拷结果）
    std::vector<aclFloat16> userHostData(M * splitArgList[0], 0);
    std::vector<aclFloat16> valueHostData(M * splitArgList[1], 0);
    std::vector<aclFloat16> queryHostData(M * splitArgList[2], 0);
    std::vector<aclFloat16> keyHostData(M * splitArgList[3], 0);
    std::vector<aclFloat16> linearOutputHostData(M * N, 0);

    const std::vector<int64_t> xShape = {M, K};
    const std::vector<int64_t> weightShape = {N, K};
    const std::vector<int64_t> biasShape = {N};
    const std::vector<int64_t> userShape = {M, splitArgList[0]};
    const std::vector<int64_t> valueShape = {M, splitArgList[1]};
    const std::vector<int64_t> queryShape = {M, splitArgList[2]};
    const std::vector<int64_t> keyShape = {M, splitArgList[3]};
    const std::vector<int64_t> linearOutputShape = {M, N};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* userDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* linearOutputDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* user = nullptr;
    aclTensor* value = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* linearOutput = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(userHostData, userShape, &userDeviceAddr, aclDataType::ACL_FLOAT16, &user);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(linearOutputHostData, linearOutputShape, &linearOutputDeviceAddr, aclDataType::ACL_FLOAT16,
                          &linearOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // split_arg_list 属性通过 aclIntArray 传入
    aclIntArray* splitArg = aclCreateIntArray(splitArgList.data(), splitArgList.size());
    CHECK_RET(splitArg != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return -1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInLinearSiluGetWorkspaceSize(x, weight, bias, splitArg, user, value, query, key, linearOutput,
                                            &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInLinearSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnInLinearSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInLinearSilu failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto userSize = M * splitArgList[0] * sizeof(aclFloat16);
    auto valueSize = M * splitArgList[1] * sizeof(aclFloat16);
    auto querySize = M * splitArgList[2] * sizeof(aclFloat16);
    auto keySize = M * splitArgList[3] * sizeof(aclFloat16);
    auto linearOutputSize = M * N * sizeof(aclFloat16);
    ret = aclrtMemcpy(userHostData.data(), userSize, userDeviceAddr, userSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy user failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(valueHostData.data(), valueSize, valueDeviceAddr, valueSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy value failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(queryHostData.data(), querySize, queryDeviceAddr, querySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy query failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(keyHostData.data(), keySize, keyDeviceAddr, keySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy key failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(linearOutputHostData.data(), linearOutputSize, linearOutputDeviceAddr, linearOutputSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy linear_output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("in_linear_silu output sample:\n");
    LOG_PRINT("  user[0,0..3]:       %.4f %.4f %.4f %.4f\n", aclFloat16ToFloat(userHostData[0]),
              aclFloat16ToFloat(userHostData[1]), aclFloat16ToFloat(userHostData[2]),
              aclFloat16ToFloat(userHostData[3]));
    LOG_PRINT("  value[0,0..3]:      %.4f %.4f %.4f %.4f\n", aclFloat16ToFloat(valueHostData[0]),
              aclFloat16ToFloat(valueHostData[1]), aclFloat16ToFloat(valueHostData[2]),
              aclFloat16ToFloat(valueHostData[3]));
    LOG_PRINT("  linear_output[0,0..3]: %.4f %.4f %.4f %.4f\n", aclFloat16ToFloat(linearOutputHostData[0]),
              aclFloat16ToFloat(linearOutputHostData[1]), aclFloat16ToFloat(linearOutputHostData[2]),
              aclFloat16ToFloat(linearOutputHostData[3]));

    aclDestroyIntArray(splitArg);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(user);
    aclDestroyTensor(value);
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(linearOutput);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(userDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(linearOutputDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
