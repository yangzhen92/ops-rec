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
 * @file test_aclnn_create_k2q_block_sparse_from_func.cpp
 * @brief 演示如何通过 aclnnCreateK2qBlockSparseFromFunc 接口调用
 *        CreateK2qBlockSparseFromFunc 算子（K2Q 反向路径）
 *
 * 算子内部为两阶段实现：第一阶段以 (b, h, q_block) 为任务粒度填充状态矩阵
 * （每个 q_block 仅读一次 func_tensor），跨核硬同步后第二阶段按列有序紧缩，
 * 输出 mask/full 块数与 block_idx 表。full 列表按 q_block 升序写入，
 * mask 列表按 q_block 升序自右写入（位置递减），保证输出顺序确定。
 *
 * 示例输入：func_tensor [1, 1, 32, 2]（n_func=2），q_len=32，kv_len=32，
 *           q_block_size=32，kv_block_size=32。
 *   每个 token 的合法 kv 区间均为 [0, 32)，覆盖整个 kv_block。
 *   期望输出：
 *     mask_block_cnt = [0]  full_block_cnt = [1]
 *     block_idx（每行 FULL 从左、MASK 从右写入）:
 *       kv_block0 -> [0]
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_create_k2q_block_sparse_from_func.h"

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
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // func_tensor: [B, H, func_q_len, n_func] = [1, 1, 32, 2]
    constexpr int64_t B = 1;
    constexpr int64_t H = 1;
    constexpr int64_t FUNC_Q_LEN = 32;
    constexpr int64_t N_FUNC = 2;
    constexpr int64_t Q_LEN = 32;
    constexpr int64_t KV_LEN = 32;
    constexpr int64_t Q_BLOCK_SIZE = 32;
    constexpr int64_t KV_BLOCK_SIZE = 32;
    constexpr int64_t NUM_Q_BLOCKS = (Q_LEN + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE;
    constexpr int64_t NUM_KV_BLOCKS = (KV_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    // 每个 q token 的合法 kv 区间 [F0, F1)，布局为连续行主序
    std::vector<int32_t> funcTensorHostData(FUNC_Q_LEN * N_FUNC);
    for (size_t i = 0; i < funcTensorHostData.size(); ++i) {
        funcTensorHostData[i] = i % 2 ? 32 : 0;
    }
    std::vector<int32_t> maskCntHostData(B * H * NUM_KV_BLOCKS, 0);
    std::vector<int32_t> fullCntHostData(B * H * NUM_KV_BLOCKS, 0);
    std::vector<int32_t> blockIdxHostData(B * H * NUM_KV_BLOCKS * NUM_Q_BLOCKS, 0);

    std::vector<int64_t> funcShape = {B, H, FUNC_Q_LEN, N_FUNC};
    std::vector<int64_t> cntShape = {B, H, NUM_KV_BLOCKS};
    std::vector<int64_t> blockIdxShape = {B, H, NUM_KV_BLOCKS, NUM_Q_BLOCKS};

    void *funcTensorDeviceAddr = nullptr;
    void *maskCntDeviceAddr = nullptr, *fullCntDeviceAddr = nullptr, *blockIdxDeviceAddr = nullptr;
    aclTensor *funcTensor = nullptr;
    aclTensor *maskBlockCnt = nullptr, *fullBlockCnt = nullptr, *blockIdx = nullptr;

    // 连续张量的 strides
    const int64_t strideB = FUNC_Q_LEN * N_FUNC;
    const int64_t strideH = FUNC_Q_LEN * N_FUNC;
    const int64_t strideQ = N_FUNC;
    const int64_t strideF = 1;

    // 后续 goto EXIT 会越过这些声明，因此全部提前到第一个 goto 之前定义
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    auto cntSize = B * H * NUM_KV_BLOCKS * sizeof(int32_t);
    auto blockIdxSize = B * H * NUM_KV_BLOCKS * NUM_Q_BLOCKS * sizeof(int32_t);

    ret = CreateAclTensor(funcTensorHostData, funcShape, &funcTensorDeviceAddr, aclDataType::ACL_INT32, &funcTensor);
    CHECK_RET(ret == ACL_SUCCESS, goto EXIT);
    ret = CreateAclTensor(maskCntHostData, cntShape, &maskCntDeviceAddr, aclDataType::ACL_INT32, &maskBlockCnt);
    CHECK_RET(ret == ACL_SUCCESS, goto EXIT);
    ret = CreateAclTensor(fullCntHostData, cntShape, &fullCntDeviceAddr, aclDataType::ACL_INT32, &fullBlockCnt);
    CHECK_RET(ret == ACL_SUCCESS, goto EXIT);
    ret = CreateAclTensor(blockIdxHostData, blockIdxShape, &blockIdxDeviceAddr, aclDataType::ACL_INT32, &blockIdx);
    CHECK_RET(ret == ACL_SUCCESS, goto EXIT);

    ret = aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize(
        funcTensor, strideB, strideH, strideQ, strideF, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, maskBlockCnt,
        fullBlockCnt, blockIdx, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCreateK2qBlockSparseFromFuncGetWorkspaceSize failed. ERROR: %d\n", ret);
              goto EXIT);

    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); goto EXIT);
    }
    ret = aclnnCreateK2qBlockSparseFromFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCreateK2qBlockSparseFromFunc failed. ERROR: %d\n", ret); goto EXIT);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); goto EXIT);

    ret = aclrtMemcpy(maskCntHostData.data(), cntSize, maskCntDeviceAddr, cntSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy mask_block_cnt failed. ERROR: %d\n", ret); goto EXIT);
    ret = aclrtMemcpy(fullCntHostData.data(), cntSize, fullCntDeviceAddr, cntSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy full_block_cnt failed. ERROR: %d\n", ret); goto EXIT);
    ret =
        aclrtMemcpy(blockIdxHostData.data(), blockIdxSize, blockIdxDeviceAddr, blockIdxSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy block_idx failed. ERROR: %d\n", ret); goto EXIT);

    LOG_PRINT("create_k2q_block_sparse_from_func mask_block_cnt:");
    for (int64_t i = 0; i < B * H * NUM_KV_BLOCKS; ++i) {
        LOG_PRINT("%d ", maskCntHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("create_k2q_block_sparse_from_func full_block_cnt:");
    for (int64_t i = 0; i < B * H * NUM_KV_BLOCKS; ++i) {
        LOG_PRINT("%d ", fullCntHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("create_k2q_block_sparse_from_func block_idx:");
    for (int64_t i = 0; i < B * H * NUM_KV_BLOCKS * NUM_Q_BLOCKS; ++i) {
        LOG_PRINT("%d ", blockIdxHostData[i]);
    }
    LOG_PRINT("\n");

EXIT:
    if (funcTensor != nullptr) {
        aclDestroyTensor(funcTensor);
    }
    if (maskBlockCnt != nullptr) {
        aclDestroyTensor(maskBlockCnt);
    }
    if (fullBlockCnt != nullptr) {
        aclDestroyTensor(fullBlockCnt);
    }
    if (blockIdx != nullptr) {
        aclDestroyTensor(blockIdx);
    }
    if (funcTensorDeviceAddr != nullptr) {
        aclrtFree(funcTensorDeviceAddr);
    }
    if (maskCntDeviceAddr != nullptr) {
        aclrtFree(maskCntDeviceAddr);
    }
    if (fullCntDeviceAddr != nullptr) {
        aclrtFree(fullCntDeviceAddr);
    }
    if (blockIdxDeviceAddr != nullptr) {
        aclrtFree(blockIdxDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
