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

#ifndef TILING_POLICY_BLOCK_SPARSE_FUNC_H
#define TILING_POLICY_BLOCK_SPARSE_FUNC_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DualExclusiveScanTilingData)
TILING_DATA_FIELD_DEF(int32_t, totalLength);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ExtractCompactTilingData)
TILING_DATA_FIELD_DEF(int32_t, nBlocks);
TILING_DATA_FIELD_DEF(int32_t, maxBlocks);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(CreateQ2kBlockSparseFromFuncTilingData)
TILING_DATA_FIELD_DEF(int32_t, strideB);
TILING_DATA_FIELD_DEF(int32_t, strideH);
TILING_DATA_FIELD_DEF(int32_t, strideQ);
TILING_DATA_FIELD_DEF(int32_t, strideF);
TILING_DATA_FIELD_DEF(int32_t, qLen);
TILING_DATA_FIELD_DEF(int32_t, kvLen);
TILING_DATA_FIELD_DEF(int32_t, nFunc);
TILING_DATA_FIELD_DEF(int32_t, qBlockSize);
TILING_DATA_FIELD_DEF(int32_t, kvBlockSize);
TILING_DATA_FIELD_DEF(int32_t, numQBlocks);
TILING_DATA_FIELD_DEF(int32_t, numKvBlocks);
TILING_DATA_FIELD_DEF(int32_t, hNum);
TILING_DATA_FIELD_DEF(int32_t, bNum);
TILING_DATA_FIELD_DEF(int32_t, totalBlocks);
TILING_DATA_FIELD_DEF(int32_t, checkQBoundary);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(CreateK2qBlockSparseFromFuncTilingData)
TILING_DATA_FIELD_DEF(int32_t, strideB);
TILING_DATA_FIELD_DEF(int32_t, strideH);
TILING_DATA_FIELD_DEF(int32_t, strideQ);
TILING_DATA_FIELD_DEF(int32_t, strideF);
TILING_DATA_FIELD_DEF(int32_t, qLen);
TILING_DATA_FIELD_DEF(int32_t, kvLen);
TILING_DATA_FIELD_DEF(int32_t, nFunc);
TILING_DATA_FIELD_DEF(int32_t, qBlockSize);
TILING_DATA_FIELD_DEF(int32_t, kvBlockSize);
TILING_DATA_FIELD_DEF(int32_t, numQBlocks);
TILING_DATA_FIELD_DEF(int32_t, numKvBlocks);
TILING_DATA_FIELD_DEF(int32_t, hNum);
TILING_DATA_FIELD_DEF(int32_t, bNum);
TILING_DATA_FIELD_DEF(int32_t, totalBlocks);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ComputeQBlockKvRangeTilingData)
TILING_DATA_FIELD_DEF(int32_t, strideB);
TILING_DATA_FIELD_DEF(int32_t, strideH);
TILING_DATA_FIELD_DEF(int32_t, strideQ);
TILING_DATA_FIELD_DEF(int32_t, strideF);
TILING_DATA_FIELD_DEF(int32_t, qLen);
TILING_DATA_FIELD_DEF(int32_t, nFunc);
TILING_DATA_FIELD_DEF(int32_t, qBlockSize);
TILING_DATA_FIELD_DEF(int32_t, numQBlocks);
TILING_DATA_FIELD_DEF(int32_t, hNum);
TILING_DATA_FIELD_DEF(int32_t, bNum);
TILING_DATA_FIELD_DEF(int32_t, totalBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DualExclusiveScan, DualExclusiveScanTilingData)
REGISTER_TILING_DATA_CLASS(ExtractCompact, ExtractCompactTilingData)
REGISTER_TILING_DATA_CLASS(CreateQ2kBlockSparseFromFunc, CreateQ2kBlockSparseFromFuncTilingData)
REGISTER_TILING_DATA_CLASS(CreateK2qBlockSparseFromFunc, CreateK2qBlockSparseFromFuncTilingData)
REGISTER_TILING_DATA_CLASS(ComputeQBlockKvRange, ComputeQBlockKvRangeTilingData)
}  // namespace optiling

#endif  // TILING_POLICY_BLOCK_SPARSE_FUNC_H
