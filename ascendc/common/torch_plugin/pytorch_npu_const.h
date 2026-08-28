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
#ifndef PYTORCH_NPU_CONST_H
#define PYTORCH_NPU_CONST_H

#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <functional>
#include <type_traits>
#include <vector>
#include <filesystem>
#include <fstream>

#include "securec.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

#define NPU_NAME_SPACE at_npu::native

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef aclTensor* (*_aclCreateTensor)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t* stride, int64_t offset, aclFormat format,
                                       const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);
typedef aclScalar* (*_aclCreateScalar)(void* value, aclDataType data_type);
typedef aclIntArray* (*_aclCreateIntArray)(const int64_t* value, uint64_t size);
typedef aclFloatArray* (*_aclCreateFloatArray)(const float* value, uint64_t size);
typedef aclBoolArray* (*_aclCreateBoolArray)(const bool* value, uint64_t size);
typedef aclTensorList* (*_aclCreateTensorList)(const aclTensor* const* value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor* tensor);
typedef int (*_aclDestroyScalar)(const aclScalar* scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray* array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray* array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray* array);
typedef int (*_aclDestroyTensorList)(const aclTensorList* array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;

#ifdef SUPPORT_950
// 兼容旧版本 ACL 头文件（如 torch_npu 打包的 third_party/acl/inc 未声明 Float8 枚举）：
// 按 CANN aclDataType 标准数值补齐，保证 kATenScalarTypeToAclDataTypeTable 在
// 编译与运行期始终拿到真实枚举值（回退 ACL_DT_UNDEFINED 会导致 fp8 输入在
// torch 适配层 ConvertType 处抛 "Float8_e4m3fn has not been supported"）。
// 头文件已声明该枚举的环境会自动选用头文件中的真实枚举值（数值一致），无需调整。
// 仅 A5(950) 生效；A2/A3 分支不引用这两个宏，行为不受影响。
#ifndef ACL_FLOAT8_E5M2
#define ACL_FLOAT8_E5M2 static_cast<aclDataType>(35)
#endif
#ifndef ACL_FLOAT8_E4M3FN
#define ACL_FLOAT8_E4M3FN static_cast<aclDataType>(36)
#endif

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)      \
    _(at::ScalarType::Byte, ACL_UINT8)                   \
    _(at::ScalarType::Char, ACL_INT8)                    \
    _(at::ScalarType::Short, ACL_INT16)                  \
    _(at::ScalarType::Int, ACL_INT32)                    \
    _(at::ScalarType::Long, ACL_INT64)                   \
    _(at::ScalarType::Half, ACL_FLOAT16)                 \
    _(at::ScalarType::Float, ACL_FLOAT)                  \
    _(at::ScalarType::Double, ACL_DOUBLE)                \
    _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED)     \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)       \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)     \
    _(at::ScalarType::Bool, ACL_BOOL)                    \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::BFloat16, ACL_BF16)                \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)        \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)        \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2)      \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN)  \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt16, ACL_UINT16)                \
    _(at::ScalarType::UInt32, ACL_UINT32)                \
    _(at::ScalarType::UInt64, ACL_UINT64)                \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

#else
#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
    _(at::ScalarType::Byte, ACL_UINT8)               \
    _(at::ScalarType::Char, ACL_INT8)                \
    _(at::ScalarType::Short, ACL_INT16)              \
    _(at::ScalarType::Int, ACL_INT32)                \
    _(at::ScalarType::Long, ACL_INT64)               \
    _(at::ScalarType::Half, ACL_FLOAT16)             \
    _(at::ScalarType::Float, ACL_FLOAT)              \
    _(at::ScalarType::Double, ACL_DOUBLE)            \
    _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED) \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)   \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL)                \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::BFloat16, ACL_BF16)            \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

#endif
#endif
