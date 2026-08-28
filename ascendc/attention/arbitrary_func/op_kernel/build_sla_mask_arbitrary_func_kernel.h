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

#ifndef ASCENDC_BUILD_SLA_MASK_ARBITRARY_FUNC_KERNEL_H_
#define ASCENDC_BUILD_SLA_MASK_ARBITRARY_FUNC_KERNEL_H_

#include <simt_api/asc_simt.h>
#include "kernel_operator.h"

namespace BuildSlaMaskArbitraryFunc {

constexpr uint32_t BLOCK_THREAD_NUM_256 = 1024;
constexpr uint32_t N_FUNC = 4;

// SLA（Semi-Local Attention）mask：架构 A 公式推导，O(1)/pos，无内层循环。
// 每个 pos_global 一行：二分定位 batch → pos_local → 闭式求 c0/c1/c2 → 写 af[q]。
// 语义见 agent_task_sla_mask.md §3.2/§3.3：
//   history: c0=min(pos+1,effK2), c1=max(effK2,pos-K1+1), c2=pos+1
//   target : c0=c1=c2=H（连自身不可见）
// 输出 af[q] = [S, S+c0, S+c1, S+c2]，全局坐标（§4）。
// 与 Python 基准 build_sla_mask_v2 唯一写法差异：用 if 合并 min/max，无分支下统一求值，等价。
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_256) inline void SimtBuildSlaMaskArbitraryFunc(
    __gm__ int32_t* __restrict__ af,             // [total_q, 4], row-major
    const __gm__ int32_t* __restrict__ offsets,  // [batchSize + 1]
    const __gm__ int32_t* __restrict__ numTargets,  // [batchSize]
    int32_t slaK1, int32_t slaK2, int32_t contextualSeqLen, int32_t totalSeqLen, int32_t batchSize,
    const uint32_t totalThread)
{
    int32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= totalSeqLen)
        return;

    // effK2 = max(slaK2, contextualSeqLen)，循环外提
    const int32_t effK2 = (slaK2 > contextualSeqLen) ? slaK2 : contextualSeqLen;

    for (; pos < totalSeqLen; pos += totalThread) {
        // 二分查找该位置所属 batch（与既有 hstu kernel 一致，O(log B)/pos）
        int32_t seqId = 0;
        {
            int32_t lo = 0, hi = batchSize;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if (offsets[mid + 1] <= pos) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            seqId = lo;
        }

        const int32_t seqStart = offsets[seqId];
        const int32_t seqLen = offsets[seqId + 1] - seqStart;
        const int32_t posLocal = pos - seqStart;
        const int32_t H = (seqLen - numTargets[seqId] < 0) ? 0 : (seqLen - numTargets[seqId]);

        const int32_t s = seqStart;
        const int32_t pos1 = posLocal + 1;
        int4 v;  // af[q] = [S, S+c0, S+c1, S+c2]
        v.x = s;
        if (posLocal < H) {
            // history: c0=min(pos+1, effK2), c1=max(effK2, pos-K1+1), c2=pos+1
            v.y = s + ((pos1 < effK2) ? pos1 : effK2);
            v.z = s + (((pos1 - slaK1) > effK2) ? (pos1 - slaK1) : effK2);
            v.w = s + pos1;
        } else {
            // target: 只见 [0, H)，连自身不可见；interval1 为空区间
            v.y = s + H;
            v.z = s + H;
            v.w = s + H;
        }
        *reinterpret_cast<__gm__ int4*>(af + pos * N_FUNC) = v;
    }
}

}  // namespace BuildSlaMaskArbitraryFunc

#endif  // ASCENDC_BUILD_SLA_MASK_ARBITRARY_FUNC_KERNEL_H_
