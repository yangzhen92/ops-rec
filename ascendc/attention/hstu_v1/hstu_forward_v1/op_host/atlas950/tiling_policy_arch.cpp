/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

// atlas950（A5）架构特化实现：
//   - TilingPolicy::TilingKeySetImpl：按 seq 长度动态选择 tile 组合
//   - GetArchWorkspaceExtraSize：A5 需额外 FP8 workspace
// 本目录仅在该 SOC 编译时被 ops_rec_collect_arch_sources 收集，与 atlasA2/ 互斥。

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tiling_policy.h"
#include "tiling_policy_define.h"
#include "hstu_kernel_tiling_key.h"

namespace HstuForward {

int64_t GetArchWorkspaceExtraSize(size_t coreNum)
{
    // A5 平台需要额外的 FP8 workspace（等于 oneCoreMidElem）
    return static_cast<int64_t>(coreNum) * VCORE_NUM_IN_ONE_AIC * BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
}

void Find4BytesShape(const TilingKeyParam& tilingKeyParam, uint32_t &tilingM, uint32_t &tilingN)
{
    constexpr uint32_t tileMList[] = {32, 64, 64, 128, 128, 256, 128};
    constexpr uint32_t tileNList[] = {32, 64, 1024, 128, 512, 256, 256};
    int idx = NO_TILING_IDX;
    int area = MAX_DIM * MAX_DIM;
    auto maxSeqLenQ = tilingKeyParam.maxSeqLenQ;
    auto maxSeqLenK = tilingKeyParam.maxSeqLenK;
    if (maxSeqLenQ > maxSeqLenK) {
        idx = (maxSeqLenK >= TILING_1024) ? 5 :
        (maxSeqLenK >= TILING_64)   ? 3 :
        (maxSeqLenK >= TILING_32)   ? 1 : 0;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }

    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area && maxSeqLenK > tileNList[i] / 2) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }

    double ratio = (double)maxSeqLenK / maxSeqLenQ;
    idx = 0;
    double delta = fabs((double)tileNList[idx] / tileMList[idx] - ratio);
    area = tileMList[idx] * tileNList[idx];
    for (int i = 1; i < TILING_SIZE; i++) {
        double tmp = fabs((double)tileNList[i] / tileMList[i] - ratio);
        if (tmp < delta || (tmp == delta && area < tileMList[i] * tileNList[i])) {
            idx = i;
            delta = tmp;
            area = tileMList[idx] * tileNList[idx];
        }
    }
    tilingM = tileMList[idx];
    tilingN = tileNList[idx];
    return;
}


void Find2BytesShape(const TilingKeyParam& tilingKeyParam, uint32_t &tilingM, uint32_t &tilingN)
{
    constexpr uint32_t tileMList[] = {32, 64, 64, 128, 128, 256, 128};
    constexpr uint32_t tileNList[] = {32, 64, 1024, 128, 512, 256, 256};
    int idx = NO_TILING_IDX;
    int area = MAX_DIM * MAX_DIM;
    auto maxSeqLenQ = tilingKeyParam.maxSeqLenQ;
    auto maxSeqLenK = tilingKeyParam.maxSeqLenK;
    if (maxSeqLenQ > maxSeqLenK) {
        if (maxSeqLenK > TILING_128) {
            idx = 5;
        } else if (maxSeqLenK > TILING_64) {
            idx = 3;
        } else if (maxSeqLenK > TILING_32) {
            idx = 1;
        } else {
            idx = 0;
        }
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area && maxSeqLenK > tileNList[i] / 2) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }

    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] * 4 && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenQ >= TILING_512 && maxSeqLenK >= TILING_1024) {
        idx = 5;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenQ >= TILING_256 && maxSeqLenK >= TILING_512) {
        idx = 4;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenK < maxSeqLenQ * 2) {
        if (maxSeqLenK >= TILING_256) {
            idx = 5;
            tilingM = tileMList[idx];
            tilingN = tileNList[idx];
            return;
        }
        for (int i = 0; i < TILING_SIZE; i++) {
            if (tileMList[i] != tileNList[i] || tileNList[i] < maxSeqLenK) {
                continue;
            }
            if (tileMList[i] * tileNList[i] < area) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    idx = 6;
    tilingM = tileMList[idx];
    tilingN = tileNList[idx];
    return;
}

void FindMatchShape(const TilingKeyParam& tilingKeyParam,
    gert::TilingContext* context, uint32_t &tilingM, uint32_t &tilingN, uint32_t &tilingDim)
{
    constexpr uint32_t tileKList[] = {64, 128, 256, 512};

    uint32_t vdim = tilingKeyParam.dimV;
    uint32_t qdim = tilingKeyParam.dimQ;
    tilingDim = *std::lower_bound(std::begin(tileKList), std::end(tileKList), std::max(qdim, vdim));
    ge::DataType qTypeGe = context->GetInputTensor(JAGGED_INPUT_INDEX_T::Q_INDEX)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        Find4BytesShape(tilingKeyParam, tilingM, tilingN);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16 || qTypeGe == ge::DataType::DT_BF16) {
        Find2BytesShape(tilingKeyParam, tilingM, tilingN);
    }
    return;
}

bool TilingPolicy::TilingKeySetImpl(gert::TilingContext* context, const TilingKeyParam& tilingKeyParam,
    uint32_t typeTilingKey)
{
    bool enableBias = tilingKeyParam.enableBias;
    bool enableDeteministic = tilingKeyParam.deterministic;
    uint32_t maskType = tilingKeyParam.maskType;
    uint32_t maskedType = maskType & 0x3;
    uint32_t tilingM = 0;
    uint32_t tilingN = 0;
    uint32_t tilingDim = 0;

    ge::DataType qTypeGe = context->GetInputTensor(JAGGED_INPUT_INDEX_T::Q_INDEX)->GetDataType();
    if ((qTypeGe == ge::DataType::DT_FLOAT8_E4M3FN) || (typeTilingKey != (JAGGED_TILING_KEY & 0x3))) {
        tilingDim = MAX_DIM;
        tilingM = MAX_TILING_DIM;
        tilingN = MAX_TILING_DIM;
    } else {
        FindMatchShape(tilingKeyParam, context, tilingM, tilingN, tilingDim);
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(maskedType, enableBias, typeTilingKey,
                                                  enableDeteministic, tilingM,
                                                  tilingN, tilingDim);
    context->SetTilingKey(tilingKey);
    return true;
}

}  // namespace HstuForward
