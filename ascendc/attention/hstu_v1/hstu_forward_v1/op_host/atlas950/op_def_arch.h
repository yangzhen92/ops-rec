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

// atlas950（A5）架构特化的 OpDef 注册项。
// 公共注册文件（hstu_dense_forward.cpp / hstu_jagged_forward.cpp / hstu_paged_forward.cpp）
// 通过 -I${OPS_REC_ARCH_DIR} 命中本文件，实现按架构目录隔离注册差异。
#ifndef HSTU_OP_DEF_ARCH_H
#define HSTU_OP_DEF_ARCH_H

// A5：q 支持 FP8 输入；输出不支持 FP8（FP8 输入映射为 FP16 输出）
#define HSTU_Q_DTYPES ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN
#define HSTU_OUT_DTYPES ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16

// A5：额外注册 ascend950 算子配置
#define HSTU_REGISTER_AICORE_CONFIG(cfg) this->AICore().AddConfig("ascend950", (cfg))

#endif  // HSTU_OP_DEF_ARCH_H
