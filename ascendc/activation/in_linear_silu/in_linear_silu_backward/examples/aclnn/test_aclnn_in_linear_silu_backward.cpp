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
 * @file test_aclnn_in_linear_silu_backward.cpp
 * @brief 演示如何通过 aclnnInLinearSiluBackward 接口调用 InLinearSiluBackward 算子
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn_in_linear_silu_backward.h"

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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 初始化 Device 和 Stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 准备 Shape 和数据
    std::vector<int64_t> xShape = {16, 64};              // 输入 x 的 shape
    std::vector<int64_t> weightShape = {256, 64};        // 权重 weight 的 shape
    std::vector<int64_t> biasShape = {256};              // 偏置 bias 的 shape
    std::vector<int64_t> gradShape = {16, 64};           // 各类梯度及输出的 shape
    std::vector<int64_t> linearOutputShape = {16, 256};  // linear_output 的 shape

    std::vector<op::fp16_t> xHostData(GetShapeSize(xShape), op::fp16_t(1.0f));
    std::vector<op::fp16_t> weightHostData(GetShapeSize(weightShape), op::fp16_t(1.0f));
    std::vector<float> biasHostData(GetShapeSize(biasShape), 0.1f);
    std::vector<op::fp16_t> userGradHostData(GetShapeSize(gradShape), op::fp16_t(1.0f));
    std::vector<op::fp16_t> valueGradHostData(GetShapeSize(gradShape), op::fp16_t(1.0f));
    std::vector<op::fp16_t> queryGradHostData(GetShapeSize(gradShape), op::fp16_t(1.0f));
    std::vector<op::fp16_t> keyGradHostData(GetShapeSize(gradShape), op::fp16_t(1.0f));
    std::vector<op::fp16_t> linearOutputHostData(GetShapeSize(linearOutputShape), op::fp16_t(0.5f));

    void *xDevAddr = nullptr, *weightDevAddr = nullptr, *biasDevAddr = nullptr;
    void *userGradDevAddr = nullptr, *valueGradDevAddr = nullptr;
    void *queryGradDevAddr = nullptr, *keyGradDevAddr = nullptr, *linearOutputDevAddr = nullptr;

    aclTensor *x = nullptr, *weight = nullptr, *bias = nullptr;
    aclTensor *userGrad = nullptr, *valueGrad = nullptr, *queryGrad = nullptr, *keyGrad = nullptr,
              *linearOutput = nullptr;

    // 创建输入及中间梯度 aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDevAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDevAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDevAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(userGradHostData, gradShape, &userGradDevAddr, aclDataType::ACL_FLOAT16, &userGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueGradHostData, gradShape, &valueGradDevAddr, aclDataType::ACL_FLOAT16, &valueGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryGradHostData, gradShape, &queryGradDevAddr, aclDataType::ACL_FLOAT16, &queryGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyGradHostData, gradShape, &keyGradDevAddr, aclDataType::ACL_FLOAT16, &keyGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(linearOutputHostData, linearOutputShape, &linearOutputDevAddr, aclDataType::ACL_FLOAT16,
                          &linearOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 准备输出梯度 Tensor (x_grad, weight_grad, bias_grad)
    std::vector<float> xGradHost(GetShapeSize(xShape), 0.0f);
    std::vector<float> weightGradHost(GetShapeSize(weightShape), 0.0f);
    std::vector<float> biasGradHost(GetShapeSize(biasShape), 0.0f);

    void *xGradDevAddr = nullptr, *weightGradDevAddr = nullptr, *biasGradDevAddr = nullptr;
    aclTensor *xGrad = nullptr, *weightGrad = nullptr, *biasGrad = nullptr;

    ret = CreateAclTensor(xGradHost, xShape, &xGradDevAddr, aclDataType::ACL_FLOAT, &xGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightGradHost, weightShape, &weightGradDevAddr, aclDataType::ACL_FLOAT, &weightGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasGradHost, biasShape, &biasGradDevAddr, aclDataType::ACL_FLOAT, &biasGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. 构造属性参数 (attr_dict, isTrans, isVardim)
    std::vector<int64_t> attrValues = {64, 64, 64, 64};
    aclIntArray* attrDict = aclCreateIntArray(attrValues.data(), attrValues.size());
    bool isTrans = false;
    bool isVardim = false;

    // 5. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段：获取 Workspace 大小
    ret = aclnnInLinearSiluBackwardGetWorkspaceSize(x, weight, bias, userGrad, valueGrad, queryGrad, keyGrad,
                                                    linearOutput, attrDict, isTrans, isVardim, xGrad, weightGrad,
                                                    biasGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInLinearSiluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据计算结果申请 Workspace 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 第二段：执行算子
    ret = aclnnInLinearSiluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInLinearSiluBackward failed. ERROR: %d\n", ret); return ret);

    // 6. 同步 Stream 等待任务结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 将计算结果从 Device 拷贝回 Host 并验证
    auto outSize = GetShapeSize(xShape);
    std::vector<float> outData(outSize, 0);
    ret = aclrtMemcpy(outData.data(), outSize * sizeof(float), xGradDevAddr, outSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("Execution succeeded! First element of x_grad: %f\n", outData[0]);

    // 8. 释放 aclTensor、aclIntArray 资源
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(userGrad);
    aclDestroyTensor(valueGrad);
    aclDestroyTensor(queryGrad);
    aclDestroyTensor(keyGrad);
    aclDestroyTensor(linearOutput);
    aclDestroyTensor(xGrad);
    aclDestroyTensor(weightGrad);
    aclDestroyTensor(biasGrad);
    aclDestroyIntArray(attrDict);

    // 9. 释放 Device 内存及清理上下文
    aclrtFree(xDevAddr);
    aclrtFree(weightDevAddr);
    aclrtFree(biasDevAddr);
    aclrtFree(userGradDevAddr);
    aclrtFree(valueGradDevAddr);
    aclrtFree(queryGradDevAddr);
    aclrtFree(keyGradDevAddr);
    aclrtFree(linearOutputDevAddr);
    aclrtFree(xGradDevAddr);
    aclrtFree(weightGradDevAddr);
    aclrtFree(biasGradDevAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
