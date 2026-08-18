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
 * @file test_aclnn_hstu_forward.cpp
 * @brief 演示如何通过 aclnnHstuDenseForward / aclnnHstuJaggedForward / aclnnHstuPagedForward
 *        接口调用 HSTU 前向融合注意力算子（normal / jagged / paged 三种数据格式）。
 *
 * 三段用例均为两段式接口：
 *   GetWorkspaceSize -> aclrtMalloc(workspace) -> aclnnHstuXxxForward 执行 -> aclrtSynchronizeDevice
 * 数据均使用 int16_t 模拟 fp16 的简单定值数据。
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_hstu_dense_forward.h"
#include "aclnn_hstu_jagged_forward.h"
#include "aclnn_hstu_paged_forward.h"

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

int TestHstuDense(aclrtStream stream)
{
    constexpr int64_t B = 1;
    constexpr int64_t S = 16;   // seq_len（max_seq_len 需覆盖 S）
    constexpr int64_t N = 2;    // head_num
    constexpr int64_t D = 32;   // head_dim（16 的倍数）
    constexpr int64_t MaxSeqLen = 16;
    constexpr int64_t MaskType = 0;  // 内置下三角 mask

    std::vector<int64_t> qkvShape = {B, S, N, D};
    auto qkvElemNum = GetShapeSize(qkvShape);
    std::vector<int16_t> qHostData(qkvElemNum);
    std::vector<int16_t> kHostData(qkvElemNum);
    std::vector<int16_t> vHostData(qkvElemNum);
    std::vector<int16_t> outHostData(qkvElemNum, 0);
    for (int64_t i = 0; i < qkvElemNum; ++i) {
        qHostData[i] = static_cast<int16_t>((i % 3 - 1) * 100);  // fp16 简单定值数据
        kHostData[i] = static_cast<int16_t>((i % 5 - 2) * 50);
        vHostData[i] = static_cast<int16_t>(i % 7);
    }

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    auto ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qkvShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHstuDenseForwardGetWorkspaceSize(q, k, v, nullptr, nullptr, MaskType, MaxSeqLen, 1.0 / MaxSeqLen,
                                                out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuDenseForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuDenseForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qkvElemNum * sizeof(int16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_dense_forward output sample [0,0,0,0..3]: %d %d %d %d\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_dense_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    return 0;
}

int TestHstuJagged(aclrtStream stream)
{
    // delta 数据构造：q 段长 [4, 6]，k/v 段长 [6, 8]（与 pytest test_hstu_jagged_delta_graph 一致）
    constexpr int64_t N = 4;            // head_num
    constexpr int64_t D = 32;           // head_dim（16 的倍数）
    constexpr int64_t TotalQ = 10;      // sum([4, 6])
    constexpr int64_t TotalK = 14;      // sum([6, 8])
    constexpr int64_t MaxSeqLenQ = 6;   // max(q 段长)
    constexpr int64_t MaxSeqLenK = 8;   // max(k 段长)
    constexpr int64_t MaskType = 0;     // 内置下三角 mask

    std::vector<int64_t> qShape = {TotalQ, N, D};
    std::vector<int64_t> kvShape = {TotalK, N, D};
    auto qElemNum = GetShapeSize(qShape);
    auto kvElemNum = GetShapeSize(kvShape);
    std::vector<int16_t> qHostData(qElemNum);
    std::vector<int16_t> kHostData(kvElemNum);
    std::vector<int16_t> vHostData(kvElemNum);
    std::vector<int16_t> outHostData(qElemNum, 0);
    for (int64_t i = 0; i < qElemNum; ++i) {
        qHostData[i] = static_cast<int16_t>((i % 3 - 1) * 100);  // fp16 简单定值数据
    }
    for (int64_t i = 0; i < kvElemNum; ++i) {
        kHostData[i] = static_cast<int16_t>((i % 5 - 2) * 50);
        vHostData[i] = static_cast<int16_t>(i % 7);
    }

    std::vector<int64_t> offsetShape = {3};
    std::vector<int32_t> seqOffsetQHostData = {0, 4, 10};
    std::vector<int32_t> seqOffsetKHostData = {0, 6, 14};

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr, *outDeviceAddr = nullptr;
    void *seqOffsetQDeviceAddr = nullptr, *seqOffsetKDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    aclTensor *seqOffsetQ = nullptr, *seqOffsetK = nullptr;
    auto ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetQHostData, offsetShape, &seqOffsetQDeviceAddr, aclDataType::ACL_INT32, &seqOffsetQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetKHostData, offsetShape, &seqOffsetKDeviceAddr, aclDataType::ACL_INT32, &seqOffsetK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // mask / attn_bias / num_context / num_target 为可选入参，不使用传 nullptr
    ret = aclnnHstuJaggedForwardGetWorkspaceSize(
        q, k, v, nullptr /*mask*/, nullptr /*attn_bias*/,
        seqOffsetQ, seqOffsetK, nullptr /*num_context*/, nullptr /*num_target*/,
        MaskType, MaxSeqLenQ, MaxSeqLenK, 1.0 / MaxSeqLenQ,
        0 /*target_group_size*/, 1.0 /*alpha*/, false /*deterministic*/,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuJaggedForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuJaggedForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuJaggedForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qElemNum * sizeof(int16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_jagged_forward output sample [0,0,0,0..3]: %d %d %d %d\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_jagged_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclDestroyTensor(seqOffsetQ);
    aclDestroyTensor(seqOffsetK);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(seqOffsetQDeviceAddr);
    aclrtFree(seqOffsetKDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    return 0;
}

int TestHstuPaged(aclrtStream stream)
{
    // paged 数据构造（与 pytest test_hstu_paged_graph 一致，已实机验证）：
    // 每个 batch：历史 token 数（已存入 kv_cache）new_hist_lens = [16, 24]，
    // 目标（新生成）token 数 num_targets = [4, 6]
    constexpr int64_t N = 4;              // head_num
    constexpr int64_t D = 32;             // head_dim（16 的倍数）
    constexpr int64_t PageSize = 16;      // 每页 KV 容量
    constexpr int64_t Batch = 2;
    constexpr int64_t Total = 50;         // (16+4) + (24+6)
    constexpr int64_t MaxSeqLen = 30;     // max(20, 30)
    constexpr int64_t NumPages = 3;       // ceil(16/16) + ceil(24/16) = 1 + 2
    constexpr int64_t MaskType = 0;       // 内置下三角 mask
    constexpr int64_t TargetGroupSize = 1;  // num_target 定义时必须 > 0（torch 扩展层同约束）

    std::vector<int64_t> qkvShape = {Total, N, D};
    std::vector<int64_t> kvCacheShape = {NumPages, 2, PageSize, N, D};
    std::vector<int64_t> offsetShape = {Batch + 1};    // seq_offset_q/k：{0, 20, 50}
    std::vector<int64_t> offsetTShape = {Batch + 1};   // seq_offset_t：{0, 4, 10}
    std::vector<int64_t> numTargetShape = {Batch};     // num_context/num_target：[2]

    auto qkvElemNum = GetShapeSize(qkvShape);
    auto kvCacheElemNum = GetShapeSize(kvCacheShape);
    std::vector<int16_t> qHostData(qkvElemNum);
    std::vector<int16_t> kHostData(qkvElemNum);
    std::vector<int16_t> vHostData(qkvElemNum);
    std::vector<int16_t> kvCacheHostData(kvCacheElemNum);
    std::vector<int16_t> outHostData(qkvElemNum, 0);
    for (int64_t i = 0; i < qkvElemNum; ++i) {
        qHostData[i] = static_cast<int16_t>((i % 3 - 1) * 100);  // fp16 简单定值数据
        kHostData[i] = static_cast<int16_t>((i % 5 - 2) * 50);
        vHostData[i] = static_cast<int16_t>(i % 7);
    }
    for (int64_t i = 0; i < kvCacheElemNum; ++i) {
        kvCacheHostData[i] = static_cast<int16_t>(i % 11 - 5);
    }

    std::vector<int32_t> seqOffsetQHostData = {0, 20, 50};
    std::vector<int32_t> seqOffsetKHostData = {0, 20, 50};
    std::vector<int32_t> seqOffsetTHostData = {0, 4, 10};
    std::vector<int32_t> pageOffsetsHostData = {0, 1, 3};   // 每个 batch 在 page_ids 中的起始页偏移
    std::vector<int32_t> pageIdsHostData = {0, 1, 2};       // 全局页号
    std::vector<int32_t> lastPageLenHostData = {16, 8};     // 每个 batch 末页有效长度（不足 page_size 时）
    std::vector<int32_t> numContextHostData = {0, 0};
    std::vector<int32_t> numTargetHostData = {4, 6};

    void *qDeviceAddr = nullptr, *kDeviceAddr = nullptr, *vDeviceAddr = nullptr, *outDeviceAddr = nullptr;
    void *kvCacheDeviceAddr = nullptr;
    void *seqOffsetQDeviceAddr = nullptr, *seqOffsetKDeviceAddr = nullptr, *seqOffsetTDeviceAddr = nullptr;
    void *pageOffsetsDeviceAddr = nullptr, *pageIdsDeviceAddr = nullptr, *lastPageLenDeviceAddr = nullptr;
    void *numContextDeviceAddr = nullptr, *numTargetDeviceAddr = nullptr;
    aclTensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    aclTensor *kvCache = nullptr;
    aclTensor *seqOffsetQ = nullptr, *seqOffsetK = nullptr, *seqOffsetT = nullptr;
    aclTensor *pageOffsets = nullptr, *pageIds = nullptr, *lastPageLen = nullptr;
    aclTensor *numContext = nullptr, *numTarget = nullptr;
    auto ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, qkvShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvCacheHostData, kvCacheShape, &kvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetQHostData, offsetShape, &seqOffsetQDeviceAddr, aclDataType::ACL_INT32, &seqOffsetQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetKHostData, offsetShape, &seqOffsetKDeviceAddr, aclDataType::ACL_INT32, &seqOffsetK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqOffsetTHostData, offsetTShape, &seqOffsetTDeviceAddr, aclDataType::ACL_INT32, &seqOffsetT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pageOffsetsHostData, offsetShape, &pageOffsetsDeviceAddr, aclDataType::ACL_INT32,
                          &pageOffsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pageIdsHostData, offsetShape, &pageIdsDeviceAddr, aclDataType::ACL_INT32, &pageIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lastPageLenHostData, numTargetShape, &lastPageLenDeviceAddr, aclDataType::ACL_INT32,
                          &lastPageLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numContextHostData, numTargetShape, &numContextDeviceAddr, aclDataType::ACL_INT32,
                          &numContext);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numTargetHostData, numTargetShape, &numTargetDeviceAddr, aclDataType::ACL_INT32, &numTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHstuPagedForwardGetWorkspaceSize(
        q, k, v, nullptr /*mask*/, nullptr /*attn_bias*/,
        seqOffsetQ, seqOffsetK, seqOffsetT,
        kvCache, pageOffsets, pageIds, lastPageLen,
        numContext, numTarget,
        MaskType, MaxSeqLen, MaxSeqLen, 1.0 / MaxSeqLen,
        TargetGroupSize, 1.0 /*alpha*/, false /*deterministic*/,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuPagedForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnHstuPagedForward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHstuPagedForward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = qkvElemNum * sizeof(int16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy output failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("hstu_paged_forward output sample [0,0,0,0..3]: %d %d %d %d\n", outHostData[0], outHostData[1],
              outHostData[2], outHostData[3]);
    LOG_PRINT("[aclnn] hstu_paged_forward example ran OK\n");

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(seqOffsetQ);
    aclDestroyTensor(seqOffsetK);
    aclDestroyTensor(seqOffsetT);
    aclDestroyTensor(pageOffsets);
    aclDestroyTensor(pageIds);
    aclDestroyTensor(lastPageLen);
    aclDestroyTensor(numContext);
    aclDestroyTensor(numTarget);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(seqOffsetQDeviceAddr);
    aclrtFree(seqOffsetKDeviceAddr);
    aclrtFree(seqOffsetTDeviceAddr);
    aclrtFree(pageOffsetsDeviceAddr);
    aclrtFree(pageIdsDeviceAddr);
    aclrtFree(lastPageLenDeviceAddr);
    aclrtFree(numContextDeviceAddr);
    aclrtFree(numTargetDeviceAddr);
    if (workspaceAddr != nullptr)
        aclrtFree(workspaceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = TestHstuDense(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuDense failed. ERROR: %d\n", ret); return ret);
    ret = TestHstuJagged(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuJagged failed. ERROR: %d\n", ret); return ret);
    ret = TestHstuPaged(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("TestHstuPaged failed. ERROR: %d\n", ret); return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
