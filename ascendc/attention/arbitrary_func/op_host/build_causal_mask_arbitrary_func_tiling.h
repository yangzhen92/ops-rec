/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_BUILD_CAUSAL_MASK_ARBITRARY_FUNC_TILING_H_
#define ASCENDC_BUILD_CAUSAL_MASK_ARBITRARY_FUNC_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BuildCausalMaskArbitraryFuncTilingData)
TILING_DATA_FIELD_DEF(int32_t, totalSeqLen);
TILING_DATA_FIELD_DEF(int32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BuildCausalMaskArbitraryFunc, BuildCausalMaskArbitraryFuncTilingData)
}  // namespace optiling

#endif  // ASCENDC_BUILD_CAUSAL_MASK_ARBITRARY_FUNC_TILING_H_
