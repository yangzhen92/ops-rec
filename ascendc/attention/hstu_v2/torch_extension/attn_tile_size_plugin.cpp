/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved. */
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <acl/acl_platform.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/interface/AclInterface.h>

#include "../common/hstu_tile_size.h"

namespace {
uint64_t GetPlatformValue(aclplatformDevInfo info)
{
    char value[64] = {};
    const auto status = aclplatformGetDeviceInfo(info, value, sizeof(value));
    TORCH_CHECK(status == ACL_SUCCESS, "failed to query Ascend platform memory, status=", status);
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0', "invalid Ascend platform memory value: ", value);
    return parsed;
}

ops_rec::hstu::CoreMemSize GetCoreMemSize()
{
    return {GetPlatformValue(ACL_PLATFORM_L0_A_SIZE), GetPlatformValue(ACL_PLATFORM_L0_B_SIZE),
            GetPlatformValue(ACL_PLATFORM_L0_C_SIZE), GetPlatformValue(ACL_PLATFORM_L1_SIZE),
            GetPlatformValue(ACL_PLATFORM_AICORE_UB_SIZE)};
}

uint32_t GetTilingHeadDim(const at::Tensor& q, const at::Tensor& v)
{
    TORCH_CHECK(q.dim() >= 1 && v.dim() >= 1, "q and v must have at least one dimension");
    const int64_t dimQk = q.size(-1);
    const int64_t dimV = v.size(-1);
    TORCH_CHECK(dimQk > 0 && dimV > 0, "q and v head dimensions must be positive");
    return static_cast<uint32_t>(std::max(dimQk, dimV));
}

std::vector<int64_t> GetFwdTileSize(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool allowDynamic)
{
    TORCH_CHECK(q.dim() >= 3 && k.dim() >= 3 && v.dim() >= 3, "q, k and v must use an attention layout");
    if (!allowDynamic) {
        const auto tile = ops_rec::hstu::GetFwdTileSize(GetTilingHeadDim(q, v));
        return {tile.m, tile.n, tile.k};
    }
    const uint32_t coreNum =
        static_cast<uint32_t>(c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE));
    const auto tile = ops_rec::hstu::GetFwdRuntimeTile(
        static_cast<uint32_t>(q.size(0)), static_cast<uint32_t>(k.size(0)), 1U, static_cast<uint32_t>(q.size(-2)),
        static_cast<uint32_t>(q.size(-1)), static_cast<uint32_t>(v.size(-1)), coreNum, GetCoreMemSize(), allowDynamic);
    return {tile.l1.m, tile.l1.n, tile.l1.k};
}

std::vector<int64_t> GetBwdTileSize(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool allowDynamic)
{
    TORCH_CHECK(q.dim() >= 3 && k.dim() >= 3 && v.dim() >= 3, "q, k and v must use an attention layout");
    if (!allowDynamic) {
        const auto tile = ops_rec::hstu::GetBwdTileSize(GetTilingHeadDim(q, v));
        return {tile.m, tile.n, tile.k};
    }
    const uint32_t coreNum =
        static_cast<uint32_t>(c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE));
    const auto tile = ops_rec::hstu::GetBwdRuntimeTile(
        static_cast<uint32_t>(q.size(0)), static_cast<uint32_t>(k.size(0)), 1U, static_cast<uint32_t>(q.size(-2)),
        static_cast<uint32_t>(q.size(-1)), static_cast<uint32_t>(v.size(-1)), coreNum, GetCoreMemSize(), allowDynamic);
    return {tile.l1.m, tile.l1.n, tile.l1.k};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("get_fwd_tile_size(Tensor q, Tensor k, Tensor v, bool allow_dynamic=True) -> int[]");
    m.def("get_bwd_tile_size(Tensor q, Tensor k, Tensor v, bool allow_dynamic=True) -> int[]");
}

TORCH_LIBRARY_IMPL(ops_rec, CompositeExplicitAutograd, m)
{
    m.impl("get_fwd_tile_size", &GetFwdTileSize);
    m.impl("get_bwd_tile_size", &GetBwdTileSize);
}
