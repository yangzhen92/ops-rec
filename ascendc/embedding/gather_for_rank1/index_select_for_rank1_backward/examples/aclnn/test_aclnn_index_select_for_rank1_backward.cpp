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
 * @file test_aclnn_index_select_for_rank1_backward.cpp
 * @brief 演示如何通过 aclnnIndexSelectForRank1Backward 接口调用 IndexSelectForRank1Backward 算子
 *
 * 注意：kernel 通过 AtomicAdd 向 grad_x 逐核累加，grad_x 必须先清零初始化
 * （等价于 torch 层的 grad_x = at::zeros_like(x)）。
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_index_select_for_rank1_backward.h"

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

    constexpr int64_t X_DIM0 = 16;    // 权重表元素个数（grad_x 长度）
    constexpr int64_t NUM_IDX = 10;   // index/grad_y 长度
    std::vector<int64_t> xShape = {X_DIM0};
    std::vector<int64_t> indexShape = {NUM_IDX};

    std::vector<float> xHostData(X_DIM0);
    std::vector<int32_t> indexHostData = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    std::vector<float> gradYHostData(NUM_IDX);
    for (int64_t i = 0; i < NUM_IDX; ++i) {
        gradYHostData[i] = static_cast<float>(i + 1) / 4.0f;
    }
    for (int64_t i = 0; i < X_DIM0; ++i) {
        xHostData[i] = static_cast<float>(i);
    }
    // grad_x 必须清零初始化：kernel 对 grad_x 做 AtomicAdd 累加
    std::vector<float> gradXHostData(X_DIM0, 0.0f);

    void* xDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* gradYDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* index = nullptr;
    aclTensor* gradY = nullptr;
    aclTensor* gradX = nullptr;
    // x 仅参与 shape 推导（grad_x 同形）与 tiling，数值不参与计算
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, indexShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, xShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnIndexSelectForRank1BackwardGetWorkspaceSize(gradY, x, index, gradX, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelectForRank1BackwardGetWorkspaceSize failed. ERROR: %d\n",
                                            ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnIndexSelectForRank1Backward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelectForRank1Backward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = X_DIM0 * sizeof(float);
    ret = aclrtMemcpy(gradXHostData.data(), outSize, gradXDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    // host 侧计算期望结果：grad_x[i] = sum(grad_y[j] where index[j] == i)
    std::vector<float> expected(X_DIM0, 0.0f);
    for (int64_t j = 0; j < NUM_IDX; ++j) {
        expected[indexHostData[j]] += gradYHostData[j];
    }
    int64_t errCnt = 0;
    for (int64_t i = 0; i < X_DIM0; ++i) {
        if (std::fabs(expected[i] - gradXHostData[i]) > 1e-6f) {
            ++errCnt;
        }
    }
    CHECK_RET(errCnt == 0, LOG_PRINT("index_select_for_rank1_backward result mismatched, errCnt=%ld\n", errCnt);
              return 1);
    LOG_PRINT("[aclnn] index_select_for_rank1_backward aclnn example PASSED\n");

    aclDestroyTensor(x);
    aclDestroyTensor(index);
    aclDestroyTensor(gradY);
    aclDestroyTensor(gradX);
    aclrtFree(xDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
