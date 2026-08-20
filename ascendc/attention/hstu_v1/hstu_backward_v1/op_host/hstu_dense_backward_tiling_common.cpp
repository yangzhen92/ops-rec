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

#include <cctype>
#include <string>

#include "register/op_def_registry.h"

#include "hstu_dense_backward_tiling_common.h"

// 这些辅助函数仅在 tiling 侧（cust_optiling）使用，proto 侧（cust_op_proto）不引用。
// 用 OP_TILING_LIB 宏隔离，避免同一符号被编进两个库而重复定义。
// GetInputLayout 同时被两侧使用，已内联到头文件。
#ifdef OP_TILING_LIB
ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char* name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->multiple = multiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % multiple != 0),
              OPS_LOG_E("", "%s must meet range[%lld %lld] and multiple of [%lld]. but get value %lld\n", name, lbound,
                        ubound, multiple, val),
              return false);
    return true;
}

bool IfMask(const int32_t& maskType, MaskType maskTypeEnum)
{
    return static_cast<int32_t>(maskTypeEnum) == maskType;
}

bool IsSameShape(const gert::Shape& shape0, const gert::Shape& shape1, int dim)
{
    if (shape0.GetDimNum() != dim || shape1.GetDimNum() != dim) {
        return false;
    }

    for (int i = 0; i < dim; i++) {
        if (shape0.GetDim(i) != shape1.GetDim(i)) {
            return false;
        }
    }

    return true;
}
#endif  // OP_TILING_LIB
