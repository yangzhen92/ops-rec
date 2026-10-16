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
============================================================================== */

#ifndef HSTU_CATLASS_FORWARD_KERNEL_TILING_HPP
#define HSTU_CATLASS_FORWARD_KERNEL_TILING_HPP

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(HstuForwardV2, ASCENDC_TPL_BOOL_DECL(HAS_RAB, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_LOCAL, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_CAUSAL, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_CONTEXT, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_TARGET, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_ARBITRARY, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(ENABLE_STREAM_K, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BLOCK_K, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 128, 256));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(HAS_RAB, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_LOCAL, 0),
                                     ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 0), ASCENDC_TPL_BOOL_SEL(IS_CONTEXT, 0),
                                     ASCENDC_TPL_BOOL_SEL(IS_TARGET, 0), ASCENDC_TPL_BOOL_SEL(IS_ARBITRARY, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(ENABLE_STREAM_K, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(BLOCK_K, ASCENDC_TPL_UI_LIST, 128, 256)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(HAS_RAB, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_LOCAL, 0),
                                     ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 1), ASCENDC_TPL_BOOL_SEL(IS_CONTEXT, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(IS_TARGET, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_ARBITRARY, 0),
                                     ASCENDC_TPL_BOOL_SEL(ENABLE_STREAM_K, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(BLOCK_K, ASCENDC_TPL_UI_LIST, 128, 256)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(HAS_RAB, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_LOCAL, 1),
                                     ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 0), ASCENDC_TPL_BOOL_SEL(IS_CONTEXT, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(IS_TARGET, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_ARBITRARY, 0),
                                     ASCENDC_TPL_BOOL_SEL(ENABLE_STREAM_K, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(BLOCK_K, ASCENDC_TPL_UI_LIST, 128, 256)));

#endif
