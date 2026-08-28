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

#ifndef FA_INFER_WITH_ARBITRARY_FUNC_TILING
#define FA_INFER_WITH_ARBITRARY_FUNC_TILING
#include "register/tilingdata_base.h"

constexpr uint32_t MAX_CORE_NUM = 64;

namespace optiling {
BEGIN_TILING_DATA_DEF(FaInferWithArbitraryFuncTilingData)
// InputParamsRegbase
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, qHeads);
TILING_DATA_FIELD_DEF(int64_t, kvHeads);
TILING_DATA_FIELD_DEF(int64_t, groupSize);
TILING_DATA_FIELD_DEF(int64_t, qSeqlen);
TILING_DATA_FIELD_DEF(int64_t, kvSeqlen);
TILING_DATA_FIELD_DEF(int64_t, embed);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);

TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsNull);
TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsKVNull);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);

TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, blockTableDim2);
TILING_DATA_FIELD_DEF(uint32_t, paBlockNumSum);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskQSeqlen);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskKvSeqlen);

// MultiCoreParamsRegbase
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, qSeqlenOuterSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CORE_NUM, bnAxisStartIdx);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM, sparseStartIdx);

uint8_t isActualSeqLengthsNull;
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FaInferWithArbitraryFunc, FaInferWithArbitraryFuncTilingData)
}  // namespace optiling

#endif
