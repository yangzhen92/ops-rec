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
 * @file test_aclnn_extract_compact.cpp
 * @brief 演示如何通过 aclnnExtractCompact 接口调用 ExtractCompact 算子
 *
 * 该算子根据每行的 mask/full 块数及其 offset，从 block_idx 查找表中提取出
 * mask_compact 与 full_compact 两个压缩后的索引缓冲。mask_compact 取每行最后
 * mask_cnt 个（逆序），full_compact 取每行前 full_cnt 个。
 *
 * 注意：mask_offset/full_offset 应由 DualExclusiveScan 算子产生（exclusive scan），
 * 本示例在 Host 侧预先计算以保持自包含。
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_extract_compact.h"

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
    constexpr int64_t H = 1;
    constexpr int64_t numBlocks = 4;
    constexpr int64_t maxBlocks = 4;
    constexpr int64_t nBlocks = B * H * numBlocks;  // 总行数 = 4
    // block_idx [B, H, numBlocks, maxBlocks] = 0..15
    std::vector<int32_t> blockIdxHostData(nBlocks * maxBlocks);
    for (int64_t i = 0; i < nBlocks * maxBlocks; ++i) {
        blockIdxHostData[i] = static_cast<int32_t>(i);
    }
    std::vector<int32_t> maskCntHostData = {1, 2, 1, 0};
    std::vector<int32_t> fullCntHostData = {2, 1, 0, 3};
    // offset 为 cnt 的 exclusive scan（此处 Host 侧预计算）
    std::vector<int32_t> maskOffsetHostData = {0, 1, 3, 4, 4};
    std::vector<int32_t> fullOffsetHostData = {0, 2, 3, 3, 6};
    std::vector<int32_t> maskCompactHostData(nBlocks * maxBlocks, 0);
    std::vector<int32_t> fullCompactHostData(nBlocks * maxBlocks, 0);

    std::vector<int64_t> blockIdxShape = {B, H, numBlocks, maxBlocks};
    std::vector<int64_t> cntShape = {B, H, numBlocks};
    std::vector<int64_t> offsetShape = {nBlocks + 1};
    std::vector<int64_t> compactShape = {nBlocks * maxBlocks};
    void *blockIdxDeviceAddr = nullptr, *maskCntDeviceAddr = nullptr, *fullCntDeviceAddr = nullptr;
    void *maskOffsetDeviceAddr = nullptr, *fullOffsetDeviceAddr = nullptr;
    void *maskCompactDeviceAddr = nullptr, *fullCompactDeviceAddr = nullptr;
    aclTensor *blockIdx = nullptr, *maskCnt = nullptr, *fullCnt = nullptr;
    aclTensor *maskOffset = nullptr, *fullOffset = nullptr, *maskCompact = nullptr, *fullCompact = nullptr;
    ret = CreateAclTensor(blockIdxHostData, blockIdxShape, &blockIdxDeviceAddr, aclDataType::ACL_INT32, &blockIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskCntHostData, cntShape, &maskCntDeviceAddr, aclDataType::ACL_INT32, &maskCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullCntHostData, cntShape, &fullCntDeviceAddr, aclDataType::ACL_INT32, &fullCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskOffsetHostData, offsetShape, &maskOffsetDeviceAddr, aclDataType::ACL_INT32, &maskOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullOffsetHostData, offsetShape, &fullOffsetDeviceAddr, aclDataType::ACL_INT32, &fullOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskCompactHostData, compactShape, &maskCompactDeviceAddr, aclDataType::ACL_INT32,
                          &maskCompact);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullCompactHostData, compactShape, &fullCompactDeviceAddr, aclDataType::ACL_INT32,
                          &fullCompact);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnExtractCompactGetWorkspaceSize(blockIdx, maskCnt, fullCnt, maskOffset, fullOffset, maskCompact,
                                              fullCompact, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExtractCompactGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnExtractCompact(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExtractCompact failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto compactSize = nBlocks * maxBlocks * sizeof(int32_t);
    ret = aclrtMemcpy(maskCompactHostData.data(), compactSize, maskCompactDeviceAddr, compactSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy mask_compact failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(fullCompactHostData.data(), compactSize, fullCompactDeviceAddr, compactSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy full_compact failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("extract_compact mask_compact:");
    for (int64_t i = 0; i < nBlocks * maxBlocks; ++i) {
        LOG_PRINT("%d ", maskCompactHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("extract_compact full_compact:");
    for (int64_t i = 0; i < nBlocks * maxBlocks; ++i) {
        LOG_PRINT("%d ", fullCompactHostData[i]);
    }
    LOG_PRINT("\n");

    aclDestroyTensor(blockIdx);
    aclDestroyTensor(maskCnt);
    aclDestroyTensor(fullCnt);
    aclDestroyTensor(maskOffset);
    aclDestroyTensor(fullOffset);
    aclDestroyTensor(maskCompact);
    aclDestroyTensor(fullCompact);
    aclrtFree(blockIdxDeviceAddr);
    aclrtFree(maskCntDeviceAddr);
    aclrtFree(fullCntDeviceAddr);
    aclrtFree(maskOffsetDeviceAddr);
    aclrtFree(fullOffsetDeviceAddr);
    aclrtFree(maskCompactDeviceAddr);
    aclrtFree(fullCompactDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
