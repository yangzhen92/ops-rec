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
 * @file test_aclnn_fa_infer_with_arbitrary_func.cpp
 * @brief 演示如何通过 aclnnFaInferWithArbitraryFunc 接口调用
 *        FaInferWithArbitraryFunc 算子
 *
 * 功能：对 q/k/v（TND 格式，headDim=128）执行带任意区间掩码的 Flash Attention。
 * 每个 query token 的合法 KV 区间由 arbitrary_func 的 4 个 INT32（两个区间
 * [af1, af2) ∪ [af3, af4)）描述，块稀疏索引（mask/full 的 cnt/offset/idx）在
 * 本示例的 host 侧预先生成。
 *
 * 示例输入：seq_len=256，heads=4，headDim=128，block_size=128，
 *           arbitrary_func 采用因果掩码：query i 合法 KV 区间为 [0, i+1)，
 *           即 af1=0、af2=i+1、af3=0、af4=0（第二个区间为空）。
 *   期望块分类：
 *     q_block0（token 0..127）：kv_block0 -> MASK，kv_block1 -> EMPTY
 *     q_block1（token 128..255）：kv_block0 -> FULL，kv_block1 -> MASK
 *
 * 编译/运行前请确保已部署 CANN 环境，并链接包含 aclnn_fa_infer_with_arbitrary_func.h 的
 * 头文件路径。
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_fa_infer_with_arbitrary_func.h"

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

// bfloat16 即 float32 高 16 位的截断表示。ACL 公开接口（acl/acl.h）未提供
// bf16 与 float 的转换函数（仅 fp16 有 aclFloatToFloat16/aclFloat16ToFloat），
// 故 host 侧用 uint16_t 存储并通过位级转换实现。
uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
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

// 判断 kv token j 对 query token i 是否合法（位于 af 两个区间并集内）
bool IsTokenValid(const std::vector<int32_t>& af, int64_t seqLen, int64_t i, int64_t j)
{
    const int32_t* a = &af[static_cast<size_t>(i) * 4];
    return (j >= a[0] && j < a[1]) || (j >= a[2] && j < a[3]);
}

// 根据 arbitrary_func 在 host 侧生成块稀疏索引：
// 每个 (b, h, q_block) 行内，kv_block 全合法 -> FULL，部分合法 -> MASK，否则 EMPTY；
// cnt 记录个数，offset 为 exclusive scan 偏移，idx 为按升序压缩后的块编号。
int BuildBlockSparse(const std::vector<int32_t>& af, int64_t seqLen, int64_t numRows, int64_t numKvBlocks,
                     int64_t blockSize, std::vector<int32_t>& maskCnt, std::vector<int32_t>& maskOffset,
                     std::vector<int32_t>& maskIdx, std::vector<int32_t>& fullCnt, std::vector<int32_t>& fullOffset,
                     std::vector<int32_t>& fullIdx)
{
    int64_t numQBlocks = (seqLen + blockSize - 1) / blockSize;
    maskCnt.assign(numRows, 0);
    fullCnt.assign(numRows, 0);
    maskIdx.clear();
    fullIdx.clear();

    for (int64_t row = 0; row < numRows; ++row) {
        int64_t qBlock = row % numQBlocks;
        int64_t qStart = qBlock * blockSize;
        int64_t qEnd = std::min(qStart + blockSize, seqLen);
        for (int64_t kb = 0; kb < numKvBlocks; ++kb) {
            int64_t kvStart = kb * blockSize;
            int64_t kvEnd = std::min(kvStart + blockSize, seqLen);
            bool allValid = true;
            bool anyValid = false;
            for (int64_t i = qStart; i < qEnd; ++i) {
                for (int64_t j = kvStart; j < kvEnd; ++j) {
                    if (IsTokenValid(af, seqLen, i, j)) {
                        anyValid = true;
                    } else {
                        allValid = false;
                    }
                }
            }
            if (allValid) {
                fullIdx.push_back(static_cast<int32_t>(kb));
                ++fullCnt[row];
            } else if (anyValid) {
                maskIdx.push_back(static_cast<int32_t>(kb));
                ++maskCnt[row];
            }
        }
    }

    maskOffset.assign(numRows + 1, 0);
    fullOffset.assign(numRows + 1, 0);
    for (int64_t row = 0; row < numRows; ++row) {
        maskOffset[row + 1] = maskOffset[row] + maskCnt[row];
        fullOffset[row + 1] = fullOffset[row] + fullCnt[row];
    }
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 满足算子约束：TND 格式（batch=1），headDim 固定为 128，block_size 为 128
    constexpr int64_t B = 1;
    constexpr int64_t HEADS = 4;
    constexpr int64_t SEQ_LEN = 256;
    constexpr int64_t HEAD_DIM = 128;
    constexpr int64_t BLOCK_SIZE = 128;
    constexpr int64_t NUM_Q_BLOCKS = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int64_t NUM_KV_BLOCKS = NUM_Q_BLOCKS;
    constexpr int64_t NUM_ROWS = B * HEADS * NUM_Q_BLOCKS;

    // q/k/v 均为 BFLOAT16，TND 格式
    const std::vector<int64_t> qShape = {SEQ_LEN, HEADS, HEAD_DIM};
    const std::vector<int64_t> kShape = {SEQ_LEN, HEADS, HEAD_DIM};
    const std::vector<int64_t> vShape = {SEQ_LEN, HEADS, HEAD_DIM};
    const std::vector<int64_t> outShape = {SEQ_LEN, HEADS, HEAD_DIM};
    const std::vector<int64_t> afShape = {B, HEADS, SEQ_LEN, 4};
    const std::vector<int64_t> cntShape = {NUM_ROWS};
    const std::vector<int64_t> offsetShape = {NUM_ROWS + 1};

    std::vector<uint16_t> qHostData(SEQ_LEN * HEADS * HEAD_DIM);
    std::vector<uint16_t> kHostData(SEQ_LEN * HEADS * HEAD_DIM);
    std::vector<uint16_t> vHostData(SEQ_LEN * HEADS * HEAD_DIM);
    std::vector<uint16_t> outHostData(SEQ_LEN * HEADS * HEAD_DIM, 0);
    for (size_t i = 0; i < qHostData.size(); ++i) {
        qHostData[i] = FloatToBf16Bits(static_cast<float>(i % 16 - 8) / 8.0f);
    }
    for (size_t i = 0; i < kHostData.size(); ++i) {
        kHostData[i] = FloatToBf16Bits(static_cast<float>(i % 32 - 16) / 16.0f);
    }
    for (size_t i = 0; i < vHostData.size(); ++i) {
        vHostData[i] = FloatToBf16Bits(static_cast<float>(i % 64 - 32) / 32.0f);
    }

    // arbitrary_func: [B, H, s_q, 4]，因果掩码 [0, i+1) ∪ [0, 0)
    std::vector<int32_t> afHostData(B * HEADS * SEQ_LEN * 4, 0);
    for (int64_t i = 0; i < SEQ_LEN; ++i) {
        afHostData[static_cast<size_t>(i) * 4 + 0] = 0;
        afHostData[static_cast<size_t>(i) * 4 + 1] = static_cast<int32_t>(i + 1);
        afHostData[static_cast<size_t>(i) * 4 + 2] = 0;
        afHostData[static_cast<size_t>(i) * 4 + 3] = 0;
    }

    // host 侧生成块稀疏索引（mask/full 的 cnt/offset/idx）
    std::vector<int32_t> maskCntHost, maskOffsetHost, maskIdxHost;
    std::vector<int32_t> fullCntHost, fullOffsetHost, fullIdxHost;
    ret = BuildBlockSparse(afHostData, SEQ_LEN, NUM_ROWS, NUM_KV_BLOCKS, BLOCK_SIZE, maskCntHost, maskOffsetHost,
                           maskIdxHost, fullCntHost, fullOffsetHost, fullIdxHost);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* afDeviceAddr = nullptr;
    void* maskCntDeviceAddr = nullptr;
    void* maskOffsetDeviceAddr = nullptr;
    void* maskIdxDeviceAddr = nullptr;
    void* fullCntDeviceAddr = nullptr;
    void* fullOffsetDeviceAddr = nullptr;
    void* fullIdxDeviceAddr = nullptr;
    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* out = nullptr;
    aclTensor* arbitraryFunc = nullptr;
    aclTensor* maskBlockCnt = nullptr;
    aclTensor* maskBlockOffset = nullptr;
    aclTensor* maskBlockIdx = nullptr;
    aclTensor* fullBlockCnt = nullptr;
    aclTensor* fullBlockOffset = nullptr;
    aclTensor* fullBlockIdx = nullptr;

    // 后续调用失败直接 return，无需释放（仅示例程序，成功路径末尾统一释放）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_BF16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_BF16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(afHostData, afShape, &afDeviceAddr, aclDataType::ACL_INT32, &arbitraryFunc);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskCntHost, cntShape, &maskCntDeviceAddr, aclDataType::ACL_INT32, &maskBlockCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskOffsetHost, offsetShape, &maskOffsetDeviceAddr, aclDataType::ACL_INT32, &maskBlockOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskIdxHost, {static_cast<int64_t>(maskIdxHost.size())}, &maskIdxDeviceAddr,
                          aclDataType::ACL_INT32, &maskBlockIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullCntHost, cntShape, &fullCntDeviceAddr, aclDataType::ACL_INT32, &fullBlockCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullOffsetHost, offsetShape, &fullOffsetDeviceAddr, aclDataType::ACL_INT32, &fullBlockOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(fullIdxHost, {static_cast<int64_t>(fullIdxHost.size())}, &fullIdxDeviceAddr,
                          aclDataType::ACL_INT32, &fullBlockIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mask / block_table 为保留输入，本算子不使用，传 nullptr
    ret = aclnnFaInferWithArbitraryFuncGetWorkspaceSize(q, k, v, nullptr, nullptr, arbitraryFunc, maskBlockCnt,
                                                        maskBlockOffset, maskBlockIdx, fullBlockCnt, fullBlockOffset,
                                                        fullBlockIdx, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFaInferWithArbitraryFuncGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnFaInferWithArbitraryFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFaInferWithArbitraryFunc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeDevice failed. ERROR: %d\n", ret); return ret);

    auto outSize = SEQ_LEN * HEADS * HEAD_DIM * sizeof(uint16_t);
    ret = aclrtMemcpy(outHostData.data(), outSize, outDeviceAddr, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy out failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("fa_infer_with_arbitrary_func mask_block_cnt:");
    for (int64_t i = 0; i < NUM_ROWS; ++i) {
        LOG_PRINT("%d ", maskCntHost[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("fa_infer_with_arbitrary_func full_block_cnt:");
    for (int64_t i = 0; i < NUM_ROWS; ++i) {
        LOG_PRINT("%d ", fullCntHost[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("fa_infer_with_arbitrary_func out[0,0,0..3]: %.4f %.4f %.4f %.4f\n", Bf16BitsToFloat(outHostData[0]),
              Bf16BitsToFloat(outHostData[1]), Bf16BitsToFloat(outHostData[2]), Bf16BitsToFloat(outHostData[3]));

    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclDestroyTensor(arbitraryFunc);
    aclDestroyTensor(maskBlockCnt);
    aclDestroyTensor(maskBlockOffset);
    aclDestroyTensor(maskBlockIdx);
    aclDestroyTensor(fullBlockCnt);
    aclDestroyTensor(fullBlockOffset);
    aclDestroyTensor(fullBlockIdx);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(afDeviceAddr);
    aclrtFree(maskCntDeviceAddr);
    aclrtFree(maskOffsetDeviceAddr);
    aclrtFree(maskIdxDeviceAddr);
    aclrtFree(fullCntDeviceAddr);
    aclrtFree(fullOffsetDeviceAddr);
    aclrtFree(fullIdxDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
