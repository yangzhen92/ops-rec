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

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "fa_infer_with_arbitrary_func_kernel_utils.h"
#include "fa_infer_with_arbitrary_func_kernel.h"

using namespace std;

extern "C" __global__ __aicore__ void fa_infer_with_arbitrary_func(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                                   GM_ADDR block_tables, GM_ADDR arbitrary_func,
                                                                   GM_ADDR mask_block_cnt, GM_ADDR mask_block_offset,
                                                                   GM_ADDR mask_block_idx, GM_ADDR full_block_cnt,
                                                                   GM_ADDR full_block_offset, GM_ADDR full_block_idx,
                                                                   GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // arbitrary mask目前仅支持以下模板参数
    FAInferTla<DTYPE_Q, true, false, true>(q, k, v, mask, block_tables, out, arbitrary_func, full_block_cnt,
                                           full_block_offset, full_block_idx, mask_block_cnt, mask_block_offset,
                                           mask_block_idx, tiling);
}
