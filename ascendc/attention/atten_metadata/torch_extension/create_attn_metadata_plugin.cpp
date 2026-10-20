/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

#include <cstdlib>
#include <string>
#define OPS_REC_CUST_VENDORS "atten_metadata"
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/interface/AclInterface.h>
#include <acl/acl_platform.h>

#include "pytorch_npu_helper.hpp"

using namespace at;

namespace {
// metadata 内存布局常量，与 op_kernel_aicpu/create_attn_metadata.h 保持一致：
//   每个 section 内 AIC/AIV core 各占 16 个 int32；head 段额外 16 个 int32。
constexpr int64_t METADATA_STRIDE = 16;
constexpr int64_t METADATA_ALIGN = 4096;

int64_t GetRuntimeL2Size()
{
    char value[64] = {};
    const auto status = aclplatformGetDeviceInfo(ACL_PLATFORM_L2_SIZE, value, sizeof(value));
    TORCH_CHECK(status == ACL_SUCCESS, "failed to query Ascend L2 size, status=", status);
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && parsed > 0, "invalid Ascend L2 size: ", value);
    return parsed;
}

int64_t DeriveBatchSize(int64_t batch_size, const c10::optional<at::Tensor>& seqused_q,
                        const c10::optional<at::Tensor>& cu_seqlens_q)
{
    if (batch_size > 0) {
        return batch_size;
    }
    if (seqused_q.has_value() && seqused_q->defined() && seqused_q->numel() > 0) {
        return seqused_q->size(0);
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q->defined() && cu_seqlens_q->numel() > 1) {
        return cu_seqlens_q->size(0) - 1;
    }
    return -1;
}

c10::optional<at::Tensor> ContiguousOpt(const c10::optional<at::Tensor>& t)
{
    if (t.has_value() && t->defined()) {
        return c10::optional<at::Tensor>(t->contiguous());
    }
    return c10::nullopt;
}
}  // namespace

at::Tensor create_attn_metadata_impl_npu(const c10::optional<at::Tensor>& cu_seqlens_q,
                                       const c10::optional<at::Tensor>& cu_seqlens_kv,
                                       const c10::optional<at::Tensor>& seqused_q,
                                       const c10::optional<at::Tensor>& seqused_kv, int64_t batch_size,
                                       int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t num_heads_q,
                                       int64_t num_heads_kv, int64_t head_dim, int64_t mask_mode, int64_t win_left,
                                       int64_t win_right, std::string layout_q, std::string layout_kv,
                                       std::string layout_out, const c10::optional<at::Tensor>& mask_block_cnt,
                                       const c10::optional<at::Tensor>& full_block_cnt, int64_t packed_row_axis,
                                       int64_t packed_mask_heads, int64_t q_block_size, int64_t kv_block_size,
                                       bool deterministic)
{
    TORCH_CHECK(num_heads_q > 0 && num_heads_kv > 0, "num_heads_q/num_heads_kv must be > 0");
    TORCH_CHECK(num_heads_q % num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv");
    int64_t batch = DeriveBatchSize(batch_size, seqused_q, cu_seqlens_q);
    TORCH_CHECK(batch > 0,
                "create_attn_metadata: batch_size must be > 0, or provide seqused_q / cu_seqlens_q to derive it.");
    const bool has_mask_count = mask_block_cnt.has_value() && mask_block_cnt->defined();
    const bool has_full_count = full_block_cnt.has_value() && full_block_cnt->defined();
    TORCH_CHECK(has_mask_count == has_full_count,
                "mask_block_cnt and full_block_cnt must be provided together");
    if (has_mask_count) {
        TORCH_CHECK(mask_block_cnt->device().type() == c10::DeviceType::PrivateUse1 &&
                        full_block_cnt->device().type() == c10::DeviceType::PrivateUse1,
                    "packed block counts must be NPU tensors");
        TORCH_CHECK(mask_block_cnt->scalar_type() == at::kInt && full_block_cnt->scalar_type() == at::kInt,
                    "packed block counts must be int32");
        TORCH_CHECK(mask_block_cnt->dim() == 3 && mask_block_cnt->sizes() == full_block_cnt->sizes(),
                    "packed block counts must have identical [B,H,outer] shapes");
        TORCH_CHECK(packed_row_axis == 0 || packed_row_axis == 1, "packed_row_axis must be 0 (Q) or 1 (K)");
        TORCH_CHECK(packed_mask_heads == 1 || packed_mask_heads == num_heads_q,
                    "packed_mask_heads must be 1 or num_heads_q");
        TORCH_CHECK(mask_block_cnt->size(0) == batch && mask_block_cnt->size(1) == packed_mask_heads,
                    "packed count batch/head geometry mismatch");
    }

    const bool packed_mode = has_mask_count;
    const int64_t aic_core_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
    const int64_t aiv_core_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
    TORCH_CHECK(aic_core_num > 0 && aiv_core_num > 0, "failed to query runtime AIC/AIV core counts");
    const int64_t direct_elems = METADATA_STRIDE + aic_core_num * METADATA_STRIDE;
    const int64_t max_sections = std::max<int64_t>(1, batch * num_heads_kv);
    const int64_t dense_elems =
        METADATA_STRIDE + max_sections * (aic_core_num + aiv_core_num) * METADATA_STRIDE;
    const int64_t required_elems = packed_mode && deterministic ? direct_elems : dense_elems;
    int64_t aligned = ((required_elems + METADATA_ALIGN - 1) / METADATA_ALIGN) * METADATA_ALIGN;

    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kInt);
    at::Tensor metadata = at::empty({aligned}, options);

    auto cu_q = ContiguousOpt(cu_seqlens_q);
    auto cu_kv = ContiguousOpt(cu_seqlens_kv);
    auto used_q = ContiguousOpt(seqused_q);
    auto used_kv = ContiguousOpt(seqused_kv);
    auto mask_cnt = ContiguousOpt(mask_block_cnt);
    auto full_cnt = ContiguousOpt(full_block_cnt);

    // ConvertTypes 以非常量左值引用接收参数，const char* 必须先落到左值再传入。
    const char* layout_q_c = layout_q.c_str();
    const char* layout_kv_c = layout_kv.c_str();
    const char* layout_out_c = layout_out.c_str();
    int64_t l2_size = GetRuntimeL2Size();

    EXEC_NPU_CMD(aclnnCreateAttnMetadata, cu_q, cu_kv, used_q, used_kv, mask_cnt, full_cnt, batch, max_seqlen_q, max_seqlen_kv, num_heads_q,
                 num_heads_kv, head_dim, mask_mode, win_left, win_right, layout_q_c, layout_kv_c, layout_out_c,
                 packed_row_axis, packed_mask_heads, l2_size, q_block_size, kv_block_size, deterministic,
                 metadata);
    return metadata;
}

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("create_attn_metadata(Tensor? cu_seqlens_q, Tensor? cu_seqlens_kv, Tensor? seqused_q, Tensor? seqused_kv, "
          "int batch_size, int max_seqlen_q, int max_seqlen_kv, int num_heads_q, int num_heads_kv, int head_dim, "
          "int mask_mode, int win_left, int win_right, str layout_q, str layout_kv, str layout_out, "
          "Tensor? mask_block_cnt=None, Tensor? full_block_cnt=None, int packed_row_axis=0, "
          "int packed_mask_heads=1, int q_block_size=0, int kv_block_size=0, bool deterministic=True) -> Tensor");
}

// 该算子的 4 个张量入参均为可选，可能全部为 None（仅靠 batch_size 等属性驱动）。
// 此时分发器无法从参数推断出 PrivateUse1(NPU) key，故注册为 CompositeExplicitAutograd
// 全后端兜底 kernel：无论入参是否带 NPU 张量都会路由到本实现（内部固定在 NPU 上分配/执行）。
TORCH_LIBRARY_IMPL(ops_rec, CompositeExplicitAutograd, m)
{
    m.impl("create_attn_metadata", &create_attn_metadata_impl_npu);
}
