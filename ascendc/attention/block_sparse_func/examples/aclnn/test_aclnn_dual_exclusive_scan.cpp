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
 * @file test_aclnn_dual_exclusive_scan.cpp
 * @brief 演示如何通过 aclnnDualExclusiveScan 接口调用 DualExclusiveScan 算子
 *
 * 该算子对 mask_cnt 与 full_cnt 并行做 exclusive scan，得到 mask_offset 与
 * full_offset（长度均为 N+1，首元素为 0）。
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_dual_exclusive_scan.h"

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

    constexpr int64_t N = 8;
    std::vector<int32_t> maskCntHostData = {1, 0, 3, 2, 1, 0, 2, 4};
    std::vector<int32_t> fullCntHostData = {2, 3, 1, 0, 4, 2, 0, 1};
    std::vector<int32_t> maskOffsetHostData(N + 1, 0);
    std::vector<int32_t> fullOffsetHostData(N + 1, 0);

    std::vector<int64_t> cntShape = {N};
    std::vector<int64_t> offsetShape = {N + 1};
    void *maskCntDeviceAddr = nullptr, *fullCntDeviceAddr = nullptr;
    void *maskOffsetDeviceAddr = nullptr, *fullOffsetDeviceAddr = nullptr;
    aclTensor *maskCnt = nullptr, *fullCnt = nullptr, *maskOffset = nullptr, *fullOffset = nullptr;
    ret = CreateAclTensor(maskCntHostData, cntShape, &maskCntDeviceAddr, aclDataType::ACL_INT32, &maskCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullCntHostData, cntShape, &fullCntDeviceAddr, aclDataType::ACL_INT32, &fullCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskOffsetHostData, offsetShape, &maskOffsetDeviceAddr, aclDataType::ACL_INT32, &maskOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullOffsetHostData, offsetShape, &fullOffsetDeviceAddr, aclDataType::ACL_INT32, &fullOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnDualExclusiveScanGetWorkspaceSize(maskCnt, fullCnt, maskOffset, fullOffset, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDualExclusiveScanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDualExclusiveScan(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDualExclusiveScan failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto offsetSize = (N + 1) * sizeof(int32_t);
    ret = aclrtMemcpy(maskOffsetHostData.data(), offsetSize, maskOffsetDeviceAddr, offsetSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy mask_offset failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(fullOffsetHostData.data(), offsetSize, fullOffsetDeviceAddr, offsetSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy full_offset failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("dual_exclusive_scan mask_offset:");
    for (int64_t i = 0; i <= N; ++i) {
        LOG_PRINT("%d ", maskOffsetHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("dual_exclusive_scan full_offset:");
    for (int64_t i = 0; i <= N; ++i) {
        LOG_PRINT("%d ", fullOffsetHostData[i]);
    }
    LOG_PRINT("\n");

    aclDestroyTensor(maskCnt);
    aclDestroyTensor(fullCnt);
    aclDestroyTensor(maskOffset);
    aclDestroyTensor(fullOffset);
    aclrtFree(maskCntDeviceAddr);
    aclrtFree(fullCntDeviceAddr);
    aclrtFree(maskOffsetDeviceAddr);
    aclrtFree(fullOffsetDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
