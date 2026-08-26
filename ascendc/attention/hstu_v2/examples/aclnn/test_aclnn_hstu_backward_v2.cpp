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
 * @file test_aclnn_hstu_backward_v2.cpp
 * @brief 演示如何通过 aclnnHstuBackwardV2 接口调用 HSTU Backward V2 算子
 *
 * HSTU Backward V2 计算 Query、Key、Value 以及 RAB (Relative Attention Bias)
 * 的梯度，v2 版本统一使用 jagged 格式输入，通过 seq_offset 管理变长序列。
 *
 * 编译依赖: libacl_ops_api.so
 * 运行前需 source set_env.sh 配置 CANN 环境
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_hstu_backward_v2.h"

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
    int32_t deviceId = 5;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 定义参数
    // 注意：heads 必须满足 1~16，dimQK/dimGV 必须是 16 的倍数，两者不能同时 > 128
    constexpr int64_t batchSize = 2;
    constexpr int64_t heads = 4;
    constexpr int64_t dimQK = 64;
    constexpr int64_t dimGV = 64;
    constexpr int64_t maxSeqLenQ = 128;
    constexpr int64_t maxSeqLenK = 128;
    constexpr int64_t totalSeqLenQ = 200;  // jagged 序列拼接后的总长度
    constexpr int64_t totalSeqLenK = 200;
    constexpr int64_t qkvSize = totalSeqLenQ * heads * dimGV;

    // 创建输入数据
    // grad: [totalSeqLenQ, heads, dimGV]
    std::vector<op::fp16_t> gradHostData(qkvSize);
    // q/k/v: [totalSeqLen, heads, dim]
    std::vector<op::fp16_t> qHostData(totalSeqLenQ * heads * dimQK);
    std::vector<op::fp16_t> kHostData(totalSeqLenK * heads * dimQK);
    std::vector<op::fp16_t> vHostData(totalSeqLenK * heads * dimGV);
    // rab: [batchSize, heads, maxSeqLenQ, maxSeqLenK]
    std::vector<op::fp16_t> rabHostData(batchSize * heads * maxSeqLenQ * maxSeqLenK);

    // 初始化输入数据
    for (int64_t i = 0; i < qkvSize; ++i) {
        gradHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i) / qkvSize);
    }
    for (int64_t i = 0; i < totalSeqLenQ * heads * dimQK; ++i) {
        qHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 100) / 100.0f);
    }
    for (int64_t i = 0; i < totalSeqLenK * heads * dimQK; ++i) {
        kHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 50) / 50.0f);
    }
    for (int64_t i = 0; i < totalSeqLenK * heads * dimGV; ++i) {
        vHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 20) / 20.0f);
    }
    for (int64_t i = 0; i < batchSize * heads * maxSeqLenQ * maxSeqLenK; ++i) {
        rabHostData[i] = static_cast<op::fp16_t>(static_cast<float>(i % 200 - 100) / 100.0f);
    }

    // seq_offset: 每个 batch 序列长度 [100, 100], offset 为 0/100/200
    std::vector<int32_t> seqOffsetQHostData = {0, 100, 200};
    std::vector<int32_t> seqOffsetKHostData = {0, 100, 200};

    // q_share: 与 q 同 shape 的 float 张量，当前算子未使用该输入
    std::vector<float> qShareHostData(totalSeqLenQ * heads * dimQK, 0.0f);

    // 定义 shape
    std::vector<int64_t> gradShape = {totalSeqLenQ, heads, dimGV};
    std::vector<int64_t> qShape = {totalSeqLenQ, heads, dimQK};
    std::vector<int64_t> kShape = {totalSeqLenK, heads, dimQK};
    std::vector<int64_t> vShape = {totalSeqLenK, heads, dimGV};
    std::vector<int64_t> seqOffsetShape = {batchSize + 1};
    std::vector<int64_t> rabShape = {batchSize, heads, maxSeqLenQ, maxSeqLenK};
    // 输出与对应输入同 shape
    std::vector<int64_t> qGradShape = qShape;
    std::vector<int64_t> kGradShape = kShape;
    std::vector<int64_t> vGradShape = vShape;
    std::vector<int64_t> rabGradShape = rabShape;

    // 输出 host 数据（初始化为 0，用于回拷时存放结果）
    std::vector<op::fp16_t> qGradHostData(GetShapeSize(qGradShape), static_cast<op::fp16_t>(0));
    std::vector<op::fp16_t> kGradHostData(GetShapeSize(kGradShape), static_cast<op::fp16_t>(0));
    std::vector<op::fp16_t> vGradHostData(GetShapeSize(vGradShape), static_cast<op::fp16_t>(0));
    std::vector<op::fp16_t> rabGradHostData(GetShapeSize(rabGradShape), static_cast<op::fp16_t>(0));

    // 创建 Tensor
    void* gradDeviceAddr = nullptr;
    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr;
    void* rabDeviceAddr = nullptr;
    void *seqOffsetQDeviceAddr = nullptr, *seqOffsetKDeviceAddr = nullptr;
    void* qShareDeviceAddr = nullptr;
    void *qGradDeviceAddr = nullptr, *kGradDeviceAddr = nullptr;
    void *vGradDeviceAddr = nullptr, *rabGradDeviceAddr = nullptr;

    aclTensor* grad = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    aclTensor* rab = nullptr;
    aclTensor *seqOffsetQ = nullptr, *seqOffsetK = nullptr;
    aclTensor* qShare = nullptr;
    aclTensor *qGrad = nullptr, *kGrad = nullptr, *vGrad = nullptr, *rabGrad = nullptr;

    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT16, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rabHostData, rabShape, &rabDeviceAddr, aclDataType::ACL_FLOAT16, &rab);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(seqOffsetQHostData, seqOffsetShape, &seqOffsetQDeviceAddr, aclDataType::ACL_INT32, &seqOffsetQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(seqOffsetKHostData, seqOffsetShape, &seqOffsetKDeviceAddr, aclDataType::ACL_INT32, &seqOffsetK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qShareHostData, qShape, &qShareDeviceAddr, aclDataType::ACL_FLOAT, &qShare);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(qGradHostData, qGradShape, &qGradDeviceAddr, aclDataType::ACL_FLOAT16, &qGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGradHostData, kGradShape, &kGradDeviceAddr, aclDataType::ACL_FLOAT16, &kGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vGradHostData, vGradShape, &vGradDeviceAddr, aclDataType::ACL_FLOAT16, &vGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rabGradHostData, rabGradShape, &rabGradDeviceAddr, aclDataType::ACL_FLOAT16, &rabGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // num_context / num_target 为可选输入，此处均传 nullptr
    // rab 已提供，因此必须同时提供 rab_grad 输出
    // scale 作为 kernel 的 dequant 标量直接下发，传 0 会导致所有梯度输出为 0，
    // 这里按惯例使用 1.0 / maxSeqLenQ
    ret = aclnnHstuBackwardV2GetWorkspaceSize(grad, q, k, v, rab, seqOffsetQ, seqOffsetK, nullptr, nullptr, qShare,
                                              maxSeqLenQ, maxSeqLenK, 1.0 / static_cast<double>(maxSeqLenQ), 1, 1.0, -1,
                                              0, qGrad, kGrad, vGrad, rabGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuBackwardV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 分配 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 执行算子
    ret = aclnnHstuBackwardV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuBackwardV2 failed. ERROR: %d\n", ret); return ret);

    // 同步等待执行完成
    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    // 拷贝输出数据回 Host
    auto qGradSize = static_cast<size_t>(GetShapeSize(qGradShape) * sizeof(op::fp16_t));
    auto kGradSize = static_cast<size_t>(GetShapeSize(kGradShape) * sizeof(op::fp16_t));
    auto vGradSize = static_cast<size_t>(GetShapeSize(vGradShape) * sizeof(op::fp16_t));
    auto rabGradSize = static_cast<size_t>(GetShapeSize(rabGradShape) * sizeof(op::fp16_t));
    ret = aclrtMemcpy(qGradHostData.data(), qGradSize, qGradDeviceAddr, qGradSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy q_grad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(kGradHostData.data(), kGradSize, kGradDeviceAddr, kGradSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy k_grad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(vGradHostData.data(), vGradSize, vGradDeviceAddr, vGradSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy v_grad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(rabGradHostData.data(), rabGradSize, rabGradDeviceAddr, rabGradSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy rab_grad failed. ERROR: %d\n", ret); return ret);

    // 打印部分输出结果
    LOG_PRINT("HSTU Backward V2 test completed successfully!\n");
    LOG_PRINT("q_grad sample [0,0,0..3]: %.4f %.4f %.4f %.4f\n", static_cast<float>(qGradHostData[0]),
              static_cast<float>(qGradHostData[1]), static_cast<float>(qGradHostData[2]),
              static_cast<float>(qGradHostData[3]));
    LOG_PRINT("k_grad sample [0,0,0..3]: %.4f %.4f %.4f %.4f\n", static_cast<float>(kGradHostData[0]),
              static_cast<float>(kGradHostData[1]), static_cast<float>(kGradHostData[2]),
              static_cast<float>(kGradHostData[3]));
    LOG_PRINT("v_grad sample [0,0,0..3]: %.4f %.4f %.4f %.4f\n", static_cast<float>(vGradHostData[0]),
              static_cast<float>(vGradHostData[1]), static_cast<float>(vGradHostData[2]),
              static_cast<float>(vGradHostData[3]));
    LOG_PRINT("rab_grad sample [0,0,0,0..3]: %.4f %.4f %.4f %.4f\n", static_cast<float>(rabGradHostData[0]),
              static_cast<float>(rabGradHostData[1]), static_cast<float>(rabGradHostData[2]),
              static_cast<float>(rabGradHostData[3]));

    // 清理资源
    aclDestroyTensor(grad);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(rab);
    aclDestroyTensor(seqOffsetQ);
    aclDestroyTensor(seqOffsetK);
    aclDestroyTensor(qShare);
    aclDestroyTensor(qGrad);
    aclDestroyTensor(kGrad);
    aclDestroyTensor(vGrad);
    aclDestroyTensor(rabGrad);

    aclrtFree(gradDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(rabDeviceAddr);
    aclrtFree(seqOffsetQDeviceAddr);
    aclrtFree(seqOffsetKDeviceAddr);
    aclrtFree(qShareDeviceAddr);
    aclrtFree(qGradDeviceAddr);
    aclrtFree(kGradDeviceAddr);
    aclrtFree(vGradDeviceAddr);
    aclrtFree(rabGradDeviceAddr);

    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
