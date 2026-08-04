/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_common.h
 * \brief ops-rec torch_extension 共用 aclnn 适配宏。
 */

#ifndef CANN_OPS_REC_ACLNN_COMMON_H
#define CANN_OPS_REC_ACLNN_COMMON_H

#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <tuple>
#include <initializer_list>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"

#define NPU_NAME_SPACE at_npu::native

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t* stride, int64_t offset, aclFormat format,
                                        const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);
using _aclCreateScalar = aclScalar* (*)(void* value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray* (*)(const int64_t* value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray* (*)(const float* value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray* (*)(const bool* value, uint64_t size);
using _aclCreateTensorList = aclTensorList* (*)(const aclTensor* const* value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor* tensor);
using _aclDestroyScalar = int (*)(const aclScalar* scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray* array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray* array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray* array);
using _aclDestroyTensorList = int (*)(const aclTensorList* array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;

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
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char* GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char* GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void* GetOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* GetOpApiLibHandler(const char* libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline std::vector<std::string> GetCustLibPath()
{
    std::string vendorPath;
    char* ascendOppPath = std::getenv("ASCEND_OPP_PATH");
    if (ascendOppPath != nullptr) {
        vendorPath = std::string(ascendOppPath) + "/vendors";
    }

    // 若 ASCEND_OPP_PATH 下找不到 config.ini，尝试从 ASCEND_CUSTOM_OPP_PATH 定位
    std::string vendorConfigFile = vendorPath + "/config.ini";
    if (!std::filesystem::exists(vendorConfigFile)) {
        char* customOppPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
        if (customOppPath != nullptr) {
            // 遍历所有 ASCEND_CUSTOM_OPP_PATH 条目，找含 vendors/config.ini 的路径
            std::string customOpp(customOppPath);
            size_t start = 0;
            while (start < customOpp.size()) {
                auto end = customOpp.find(':', start);
                std::string entry =
                    (end != std::string::npos) ? customOpp.substr(start, end - start) : customOpp.substr(start);
                // 尝试 entry 自身或其父目录下的 vendors/config.ini
                std::string candidate = entry + "/vendors/config.ini";
                if (std::filesystem::exists(candidate)) {
                    vendorConfigFile = candidate;
                    vendorPath = entry + "/vendors";
                    break;
                }
                if (end == std::string::npos)
                    break;
                start = end + 1;
            }
        }
    }

    if (!std::filesystem::exists(vendorConfigFile)) {
        ASCEND_LOGW("config.ini not found under ASCEND_OPP_PATH or ASCEND_CUSTOM_OPP_PATH.");
        return std::vector<std::string>();
    }

    std::ifstream ifs(vendorConfigFile);
    std::string line;
    bool found = false;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            found = true;
            break;
        }
    }

    if (!found) {
        ASCEND_LOGW("config.ini %s missing 'load_priority=' entry.", vendorConfigFile.c_str());
        return std::vector<std::string>();
    }

    std::string head = "load_priority=";
    line.erase(0, head.length());

    std::vector<std::string> paths;
    size_t pos = 0;
    while ((pos = line.find(',')) != std::string::npos) {
        paths.push_back(vendorPath + "/" + line.substr(0, pos) + "/op_api/lib");
        line.erase(0, pos + 1);
    }
    paths.push_back(vendorPath + "/" + line + "/op_api/lib");
    return paths;
}

inline void* GetOpApiFuncAddr(const char* apiName)
{
    const std::vector<std::string> gCustomLibPath = GetCustLibPath();
    for (const auto& path : gCustomLibPath) {
        auto custOpApiLib = path + "/" + GetCustOpApiLibName();
        if (custOpApiLib.empty()) {
            break;
        }
        auto custOpApiHandler = GetOpApiLibHandler(custOpApiLib.c_str());
        if (custOpApiHandler != nullptr) {
            auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
            if (funcAddr != nullptr) {
                return funcAddr;
            }
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline aclTensor* ConvertType(const at::Tensor& at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported");
    std::vector<int64_t> storageDims;
    auto itemsize = at_tensor.itemsize();
    if (itemsize == 0) {
        AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
        return nullptr;
    }
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    }

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
            break;
    }

    auto acl_tensor =
        aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclScalar* ConvertType(const at::Scalar& at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported");
    return nullptr;
}

inline aclIntArray* ConvertType(const at::IntArrayRef& at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    return aclCreateIntArray(at_array.data(), at_array.size());
}

// Pass-through for non-tensor types used in ACLNN_CMD macro
inline uint64_t* ConvertType(uint64_t* v)
{
    return v;
}
inline aclOpExecutor** ConvertType(aclOpExecutor** v)
{
    return v;
}

inline void Release(aclTensor* p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar* p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray* p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }
    aclDestroyIntArray(p);
}

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

uint64_t CalcHashId();
typedef int (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);

#define ACLNN_CMD(aclnn_api, ...)                                                                                   \
    do {                                                                                                            \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");               \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                             \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                 \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                             \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                      \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",            \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),       \
                    "not found.");                                                                                  \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                             \
        uint64_t workspace_size = 0;                                                                                \
        uint64_t* workspace_size_addr = &workspace_size;                                                            \
        aclOpExecutor* executor = nullptr;                                                                          \
        aclOpExecutor** executor_addr = &executor;                                                                  \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                 \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);         \
        if (initMemFunc) {                                                                                          \
            initMemFunc(nullptr, false);                                                                            \
        }                                                                                                           \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                      \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);          \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                       \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());            \
        void* workspace_addr = nullptr;                                                                             \
        at::Tensor workspace_tensor;                                                                                \
        if (workspace_size != 0) {                                                                                  \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());                 \
            workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));         \
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                                  \
        }                                                                                                           \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() mutable -> int { \
            (void)converted_params;                                                                                 \
            typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                           \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                       \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                         \
            ReleaseConvertTypes(converted_params);                                                                  \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                       \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                 \
            if (releaseMemFunc) {                                                                                   \
                releaseMemFunc(nullptr, false);                                                                     \
            }                                                                                                       \
            return api_ret;                                                                                         \
        };                                                                                                          \
        at_npu::native::OpCommand cmd;                                                                              \
        cmd.Name(#aclnn_api);                                                                                       \
        cmd.SetCustomHandler(acl_call);                                                                             \
        cmd.Run();                                                                                                  \
        if (unInitMemFunc) {                                                                                        \
            unInitMemFunc(nullptr, false);                                                                          \
        }                                                                                                           \
    } while (false)

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>)
{
    typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

#endif  // CANN_OPS_REC_ACLNN_COMMON_H
