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

// atlasA2/A3 架构特化实现：
//   - TilingPolicy::TilingKeySetImpl：固定 tiling key（不按 seq 长度选 tile）
//   - GetArchWorkspaceExtraSize：A2/A3 无额外 workspace
// 本目录仅在该 SOC 编译时被 ops_rec_collect_arch_sources 收集，与 atlas950/ 互斥。

#include <cstdint>

#include "tiling_policy.h"
#include "tiling_policy_define.h"
#include "hstu_kernel_tiling_key.h"

namespace HstuForward {

int64_t GetArchWorkspaceExtraSize(size_t coreNum)
{
    // A2/A3 平台无额外 workspace 需求
    (void)coreNum;
    return 0;
}

bool TilingPolicy::TilingKeySetImpl(gert::TilingContext* context, const TilingKeyParam& tilingKeyParam,
    uint32_t typeTilingKey)
{
    bool enableBias = tilingKeyParam.enableBias;
    bool enableDeteministic = tilingKeyParam.deterministic;
    uint32_t maskType = tilingKeyParam.maskType;
    uint32_t maskedType = maskType & 0x3;
    // 组合tiling key：

    const uint64_t tilingKey = GET_TPL_TILING_KEY(maskedType, enableBias,
                                                  typeTilingKey, enableDeteministic, MAX_TILING_DIM,
                                                  MAX_TILING_DIM, MAX_DIM);
    context->SetTilingKey(tilingKey);

    return true;
}

}  // namespace HstuForward
