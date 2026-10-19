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
 * @file test_aclnn_disentangle_attention.cpp
 * @brief 演示如何通过 aclnnDisentangleAttention 接口调用 DisentangleAttention 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn_disentangle_attention.h"

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

    // 定义参数
    // 注意：根据约束，seq_len 必须为 256，head_dim 必须是 16 的倍数
    constexpr int64_t batch = 2;
    constexpr int64_t heads = 4;
    constexpr int64_t seq_len = 256;  // 算子约束：seq_len 必须固定为 256
    constexpr int64_t head_dim = 64;  // 算子约束：必须是 16 的倍数

    // 创建输入数据
    // query_layer, key_layer, value_layer: [batch, heads, seq_len, head_dim]
    int64_t qkv_size = batch * heads * seq_len * head_dim;
    std::vector<op::fp16_t> queryHostData(qkv_size);
    std::vector<op::fp16_t> keyHostData(qkv_size);
    std::vector<op::fp16_t> valueHostData(qkv_size);

    // pos_key_layer, pos_query_layer: [2*seq_len, heads, head_dim]
    int64_t pos_size = 2 * seq_len * heads * head_dim;
    std::vector<op::fp16_t> posKeyHostData(pos_size);
    std::vector<op::fp16_t> posQueryHostData(pos_size);

    // relative_pos: [1, 1, seq_len, seq_len]
    int64_t relative_pos_size = seq_len * seq_len;
    std::vector<int64_t> relativePosHostData(relative_pos_size);

    // atten_mask: [batch, 1, seq_len, seq_len]
    int64_t mask_size = batch * seq_len * seq_len;
    std::vector<op::fp16_t> attenMaskHostData(mask_size);

    // 输出数据
    std::vector<op::fp16_t> outputHostData(qkv_size, 0);
    std::vector<op::fp16_t> probsHostData(batch * heads * seq_len * seq_len, 0);
    std::vector<op::fp16_t> weightsHostData(batch * heads * seq_len * seq_len, 0);

    // 初始化输入数据
    for (int i = 0; i < qkv_size; ++i) {
        queryHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i) / qkv_size);
        keyHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 100) / 100.0f);
        valueHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 50) / 50.0f);
    }
    for (int i = 0; i < pos_size; ++i) {
        posKeyHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 30) / 30.0f);
        posQueryHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 20) / 20.0f);
    }
    for (int i = 0; i < relative_pos_size; ++i) {
        relativePosHostData[i] = i % (2 * seq_len) - seq_len;  // 相对位置编码
    }
    for (int i = 0; i < mask_size; ++i) {
        attenMaskHostData[i] = static_cast<op::fp16_t>(0.0f);  // 无 mask
    }

    // 定义 shape
    std::vector<int64_t> qkv_shape = {batch, heads, seq_len, head_dim};
    std::vector<int64_t> pos_shape = {2 * seq_len, heads, head_dim};
    std::vector<int64_t> relative_pos_shape = {1, 1, seq_len, seq_len};
    std::vector<int64_t> mask_shape = {batch, 1, seq_len, seq_len};

    // 创建 Tensor
    void *queryDeviceAddr = nullptr, *keyDeviceAddr = nullptr, *valueDeviceAddr = nullptr;
    void *posKeyDeviceAddr = nullptr, *posQueryDeviceAddr = nullptr;
    void *relativePosDeviceAddr = nullptr, *attenMaskDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr, *probsDeviceAddr = nullptr, *weightsDeviceAddr = nullptr;

    aclTensor *query = nullptr, *key = nullptr, *value = nullptr;
    aclTensor *posKey = nullptr, *posQuery = nullptr;
    aclTensor *relativePos = nullptr, *attenMask = nullptr;
    aclTensor *output = nullptr, *probs = nullptr, *weights = nullptr;

    ret = CreateAclTensor(queryHostData, qkv_shape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkv_shape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, qkv_shape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(posKeyHostData, pos_shape, &posKeyDeviceAddr, aclDataType::ACL_FLOAT16, &posKey);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(posQueryHostData, pos_shape, &posQueryDeviceAddr, aclDataType::ACL_FLOAT16, &posQuery);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(relativePosHostData, relative_pos_shape, &relativePosDeviceAddr, aclDataType::ACL_INT64,
                          &relativePos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenMaskHostData, mask_shape, &attenMaskDeviceAddr, aclDataType::ACL_FLOAT16, &attenMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputHostData, qkv_shape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> probs_shape = {batch, heads, seq_len, seq_len};
    ret = CreateAclTensor(probsHostData, probs_shape, &probsDeviceAddr, aclDataType::ACL_FLOAT16, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> weights_shape = {batch, heads, seq_len, seq_len};
    ret = CreateAclTensor(weightsHostData, weights_shape, &weightsDeviceAddr, aclDataType::ACL_FLOAT16, &weights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 获取 workspace 大小
    int32_t pos_attr_type = 2;  // c2p|p2c 模式
    float score_scale = 1.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnDisentangleAttentionGetWorkspaceSize(query, key, value, posKey, posQuery, relativePos, attenMask,
                                                    pos_attr_type, score_scale, output, probs, weights, &workspaceSize,
                                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDisentangleAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 执行算子
    ret = aclnnDisentangleAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDisentangleAttention failed. ERROR: %d\n", ret); return ret);

    // 同步等待执行完成
    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    // 拷贝输出数据回 Host
    auto output_size = qkv_size * sizeof(op::fp16_t);
    ret = aclrtMemcpy(outputHostData.data(), output_size, outputDeviceAddr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    auto probs_size = batch * heads * seq_len * seq_len * sizeof(op::fp16_t);
    ret = aclrtMemcpy(probsHostData.data(), probs_size, probsDeviceAddr, probs_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy probs failed. ERROR: %d\n", ret); return ret);

    auto weights_size = batch * heads * seq_len * seq_len * sizeof(op::fp16_t);
    ret = aclrtMemcpy(weightsHostData.data(), weights_size, weightsDeviceAddr, weights_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy weights failed. ERROR: %d\n", ret); return ret);

    // 打印部分输出结果
    LOG_PRINT("Disentangle Attention test completed successfully!\n");
    LOG_PRINT("Output shape: [%ld, %ld, %ld, %ld]\n", batch, heads, seq_len, head_dim);
    LOG_PRINT("Output sample [0,0,0,0..3]: %.4f %.4f %.4f %.4f\n", static_cast<float>(outputHostData[0]),
              static_cast<float>(outputHostData[1]), static_cast<float>(outputHostData[2]),
              static_cast<float>(outputHostData[3]));

    // 清理资源
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(posKey);
    aclDestroyTensor(posQuery);
    aclDestroyTensor(relativePos);
    aclDestroyTensor(attenMask);
    aclDestroyTensor(output);
    aclDestroyTensor(probs);
    aclDestroyTensor(weights);

    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(posKeyDeviceAddr);
    aclrtFree(posQueryDeviceAddr);
    aclrtFree(relativePosDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(weightsDeviceAddr);

    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
