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
 * @file test_aclnn_reverse_sequence.cpp
 * @brief 演示如何通过 aclnnReverseSequenceCust 接口调用 ReverseSequenceCust 算子
 *
 * 注意：kernel 仅将每个 batch 前 seq_len 行逆序写入 output，其余行不会被修改。
 * 因此 output 需要先用 input 的数据初始化（等价于 torch 层的 output = input.clone()）。
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_reverse_sequence_cust.h"

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

    constexpr int64_t BS = 4;
    constexpr int64_t MAX_SEQ_LEN = 8;
    constexpr int64_t DATA_DIM = 16;
    constexpr int64_t DATA_NUM = BS * MAX_SEQ_LEN * DATA_DIM;
    std::vector<float> inputHostData(DATA_NUM);
    std::vector<int64_t> seqLengthsHostData = {2, 8, 0, 5};
    // output 先用 input 初始化：kernel 只会覆写每个 batch 前 seq_len 行（逆序位置）
    std::vector<float> outHostData(DATA_NUM, 0);
    for (int64_t i = 0; i < DATA_NUM; ++i) {
        inputHostData[i] = static_cast<float>(i % 97) / 97.0f;
    }

    std::vector<int64_t> inputShape = {BS, MAX_SEQ_LEN, DATA_DIM};
    std::vector<int64_t> seqLengthsShape = {BS};
    void* inputDeviceAddr = nullptr;
    void* seqLengthsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* seqLengths = nullptr;
    aclTensor* out = nullptr;
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqLengthsHostData, seqLengthsShape, &seqLengthsDeviceAddr, aclDataType::ACL_INT64,
                          &seqLengths);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 与 torch 层 output = input.clone() 等价：先把 input 数据写入 output buffer
    ret = CreateAclTensor(inputHostData, inputShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnReverseSequenceCustGetWorkspaceSize(input, seqLengths, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnReverseSequenceCustGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnReverseSequenceCust(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReverseSequenceCust failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = DATA_NUM * sizeof(float);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    // host 侧计算期望结果并逐元素校验
    int64_t errCnt = 0;
    for (int64_t b = 0; b < BS; ++b) {
        int64_t len = seqLengthsHostData[b];
        for (int64_t s = 0; s < MAX_SEQ_LEN; ++s) {
            int64_t srcRow = (s < len) ? (len - 1 - s) : s;
            for (int64_t d = 0; d < DATA_DIM; ++d) {
                float expected = inputHostData[(b * MAX_SEQ_LEN + srcRow) * DATA_DIM + d];
                float actual = outHostData[(b * MAX_SEQ_LEN + s) * DATA_DIM + d];
                if (std::fabs(expected - actual) > 1e-6f) {
                    ++errCnt;
                }
            }
        }
    }
    CHECK_RET(errCnt == 0, LOG_PRINT("reverse_sequence result mismatched, errCnt=%ld\n", errCnt); return 1);
    LOG_PRINT("[aclnn] reverse_sequence aclnn example PASSED\n");

    aclDestroyTensor(input);
    aclDestroyTensor(seqLengths);
    aclDestroyTensor(out);
    aclrtFree(inputDeviceAddr);
    aclrtFree(seqLengthsDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
