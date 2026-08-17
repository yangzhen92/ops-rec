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
 * @file test_aclnn_compute_q_block_kv_range.cpp
 * @brief 演示如何通过 aclnnComputeQBlockKvRange 接口调用 ComputeQBlockKvRange 算子
 *
 * 为每个 (b, h, q_block) 预计算其覆盖的 kv 区间下界与上界，输出 q_block_kv_min /
 * q_block_kv_max（shape [B, H, num_q_blocks]），供 CreateK2qBlockSparseFromFunc
 * 做 kv-range 优化。
 *
 * 示例输入：func_tensor [1, 1, 4, 2]（n_func=2），q_len=8，q_block_size=4。
 *   token 区间（[F0, F1)）：
 *     token0: [0, 6]   token1: [0, 6]   token2: [2, 6]   token3: [2, 6]
 *   期望输出：
 *     q_block_kv_min = [0, 2]
 *     q_block_kv_max = [6, 6]
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_compute_q_block_kv_range.h"

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

    // func_tensor: [B, H, func_q_len, n_func] = [1, 1, 8, 2]
    constexpr int64_t B = 1;
    constexpr int64_t H = 1;
    constexpr int64_t FUNC_Q_LEN = 8;
    constexpr int64_t N_FUNC = 2;
    constexpr int64_t Q_LEN = 8;
    constexpr int64_t Q_BLOCK_SIZE = 4;
    constexpr int64_t NUM_Q_BLOCKS = (Q_LEN + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE;

    // 每个 q token 的合法 kv 区间 [F0, F1)，布局为连续行主序
    std::vector<int32_t> funcTensorHostData = {
        0, 8,  // token0: [0, 8)
        0, 8,  // token1: [0, 8)
        2, 8,  // token2: [2, 8)
        2, 8,  // token3: [2, 8)
        4, 8,  // token4: [4, 8)
        4, 8,  // token5: [4, 8)
        0, 4,  // token6: [0, 4)
        0, 4,  // token7: [0, 4)
    };

    std::vector<int32_t> kvMinHostData(NUM_Q_BLOCKS, 0);
    std::vector<int32_t> kvMaxHostData(NUM_Q_BLOCKS, 0);
    std::vector<int64_t> funcShape = {B, H, FUNC_Q_LEN, N_FUNC};
    std::vector<int64_t> kvRangeShape = {B, H, NUM_Q_BLOCKS};

    void *funcTensorDeviceAddr = nullptr, *kvMinDeviceAddr = nullptr, *kvMaxDeviceAddr = nullptr;
    aclTensor *funcTensor = nullptr, *qBlockKvMin = nullptr, *qBlockKvMax = nullptr;
    ret = CreateAclTensor(funcTensorHostData, funcShape, &funcTensorDeviceAddr, aclDataType::ACL_INT32, &funcTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvMinHostData, kvRangeShape, &kvMinDeviceAddr, aclDataType::ACL_INT32, &qBlockKvMin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvMaxHostData, kvRangeShape, &kvMaxDeviceAddr, aclDataType::ACL_INT32, &qBlockKvMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 连续张量的 strides
    const int64_t strideB = FUNC_Q_LEN * N_FUNC;
    const int64_t strideH = FUNC_Q_LEN * N_FUNC;
    const int64_t strideQ = N_FUNC;
    const int64_t strideF = 1;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnComputeQBlockKvRangeGetWorkspaceSize(funcTensor, strideB, strideH, strideQ, strideF, Q_LEN,
                                                    Q_BLOCK_SIZE, qBlockKvMin, qBlockKvMax, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComputeQBlockKvRangeGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnComputeQBlockKvRange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComputeQBlockKvRange failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto kvRangeSize = NUM_Q_BLOCKS * sizeof(int32_t);
    ret = aclrtMemcpy(kvMinHostData.data(), kvRangeSize, kvMinDeviceAddr, kvRangeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy q_block_kv_min failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(kvMaxHostData.data(), kvRangeSize, kvMaxDeviceAddr, kvRangeSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy q_block_kv_max failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("compute_q_block_kv_range q_block_kv_min:");
    for (int64_t i = 0; i < NUM_Q_BLOCKS; ++i) {
        LOG_PRINT("%d ", kvMinHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("compute_q_block_kv_range q_block_kv_max:");
    for (int64_t i = 0; i < NUM_Q_BLOCKS; ++i) {
        LOG_PRINT("%d ", kvMaxHostData[i]);
    }
    LOG_PRINT("\n");

    aclDestroyTensor(funcTensor);
    aclDestroyTensor(qBlockKvMin);
    aclDestroyTensor(qBlockKvMax);
    aclrtFree(funcTensorDeviceAddr);
    aclrtFree(kvMinDeviceAddr);
    aclrtFree(kvMaxDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
