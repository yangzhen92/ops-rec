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

#ifndef INDEX_SELECT_FOR_RANK1_BACKWARD_TILING_H
#define INDEX_SELECT_FOR_RANK1_BACKWARD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexSelectForRank1BackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLen);        // index 元素总数（本次 backward 要散射的梯度个数）
TILING_DATA_FIELD_DEF(int64_t, xDim0);           // 权重表 x 的元素个数（grad_x 的长度）
TILING_DATA_FIELD_DEF(int64_t, baseLen);         // 单核均摊的 index 元素个数
TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);  // 前 tailSplitIndex 个核各多处理 1 个元素
TILING_DATA_FIELD_DEF(int64_t, stride);          // 单次搬入 UB 的 index/grad_y 元素个数上限
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexSelectForRank1Backward, IndexSelectForRank1BackwardTilingData)
}  // namespace optiling
#endif  // INDEX_SELECT_FOR_RANK1_BACKWARD_TILING_H
