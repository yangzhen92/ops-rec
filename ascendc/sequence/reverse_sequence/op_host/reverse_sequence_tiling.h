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

#ifndef REVERSE_SEQUENCE_TILING_H
#define REVERSE_SEQUENCE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReverseSequenceTilling)

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, maxSeqLen);
TILING_DATA_FIELD_DEF(int64_t, dataDim);
TILING_DATA_FIELD_DEF(int64_t, left);
TILING_DATA_FIELD_DEF(int64_t, handleTotalCount);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerTile);    // 单个 UB tile 容纳的行数（DataCopyPad blockCount 上限内）
TILING_DATA_FIELD_DEF(uint32_t, rowAlignCount);  // 单行按 32B 对齐后的元素个数，即 UB 内的行 stride

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReverseSequenceCust, ReverseSequenceTilling)
}  // namespace optiling
#endif
