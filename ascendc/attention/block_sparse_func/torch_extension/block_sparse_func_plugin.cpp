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

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

void validate_compact_no_sync_inputs(const Tensor& mask_block_cnt, const Tensor& full_block_cnt,
                                     const Tensor& block_idx)
{
    // 检查维度
    check_tensor_dim(mask_block_cnt, 3, "mask_block_cnt");
    check_tensor_dim(full_block_cnt, 3, "full_block_cnt");
    check_tensor_dim(block_idx, 4, "block_idx");

    // 检查设备
    check_tensor_npu_device({mask_block_cnt, full_block_cnt, block_idx},
                            {"mask_block_cnt", "full_block_cnt", "block_idx"});

    // 检查数据类型
    TORCH_CHECK(mask_block_cnt.dtype() == at::kInt, "mask_block_cnt must be int32");
    TORCH_CHECK(full_block_cnt.dtype() == at::kInt, "full_block_cnt must be int32");
    TORCH_CHECK(block_idx.dtype() == at::kInt, "block_idx must be int32");

    // 检查形状一致性
    auto mask_shape = mask_block_cnt.sizes();
    auto full_shape = full_block_cnt.sizes();
    auto block_shape = block_idx.sizes();

    TORCH_CHECK(mask_shape == full_shape, "mask_block_cnt and full_block_cnt must have the same shape");

    TORCH_CHECK(mask_shape[0] == block_shape[0] && mask_shape[1] == block_shape[1] && mask_shape[2] == block_shape[2],
                "block_idx shape mismatch");
}

/**
 * compact_no_sync算子的NPU实现
 * @param mask_block_cnt [B, H, num_blocks] 每个query块的mask块数
 * @param full_block_cnt [B, H, num_blocks] 每个query块的full块数
 * @param block_idx [B, H, num_blocks, max_blocks] 块索引
 * @return 元组包含(mask_block_cnt, mask_offset, mask_compact, full_block_cnt, full_offset, full_compact)
 */
std::vector<Tensor> compact_no_sync_impl_npu(const Tensor& mask_block_cnt, const Tensor& full_block_cnt,
                                             const Tensor& block_idx)
{
    validate_compact_no_sync_inputs(mask_block_cnt, full_block_cnt, block_idx);

    // contiguous处理
    auto maskBlockCntConti = mask_block_cnt.contiguous();
    auto fullBlockCntConti = full_block_cnt.contiguous();
    auto blockIdxConti = block_idx.contiguous();

    // 获取形状参数
    int64_t B = mask_block_cnt.size(0);
    int64_t H = mask_block_cnt.size(1);
    int64_t numBlocks = mask_block_cnt.size(2);
    int64_t maxBlocks = block_idx.size(3);

    int64_t nFlat = B * H * numBlocks;

    // 分配输出
    at::Tensor maskOffset = at::empty({nFlat + 1}, mask_block_cnt.options());
    at::Tensor fullOffset = at::empty({nFlat + 1}, full_block_cnt.options());
    at::Tensor maskCompact = at::full({nFlat * maxBlocks}, -1, mask_block_cnt.options());
    at::Tensor fullCompact = at::full({nFlat * maxBlocks}, -1, block_idx.options());

    // 调用 dual_exclusive_scan kernel
    EXEC_NPU_CMD(aclnnDualExclusiveScan, maskBlockCntConti, fullBlockCntConti, maskOffset, fullOffset);

    // 调用 extract_compact kernel
    EXEC_NPU_CMD(aclnnExtractCompact, blockIdxConti, maskBlockCntConti, fullBlockCntConti, maskOffset, fullOffset,
                 maskCompact, fullCompact);

    // 返回6-tuple
    return {maskBlockCntConti, maskOffset, maskCompact, fullBlockCntConti, fullOffset, fullCompact};
}

namespace {
/**
 * compact_no_sync算子的元函数实现
 * @param mask_block_cnt [B, H, num_blocks] 每个query块的mask块数
 * @param full_block_cnt [B, H, num_blocks] 每个query块的full块数
 * @param block_idx [B, H, num_blocks, max_blocks] 块索引
 * @return 元组包含(mask_block_cnt, mask_offset, mask_compact, full_block_cnt, full_offset, full_compact)
 */
tensor_list compact_no_sync_meta(const Tensor& mask_block_cnt, const Tensor& full_block_cnt, const Tensor& block_idx)
{
    TORCH_CHECK(mask_block_cnt.dim() == 3, "mask_block_cnt must be 3D");
    TORCH_CHECK(full_block_cnt.dim() == 3, "full_block_cnt must be 3D");
    TORCH_CHECK(block_idx.dim() == 4, "block_idx must be 4D");

    // mask_block_cnt/full_block_cnt 原样返回
    auto mask_cnt = at::empty_like(mask_block_cnt, mask_block_cnt.options());
    auto full_cnt = at::empty_like(full_block_cnt, full_block_cnt.options());

    // offset 是 cnt 的累加前缀和，shape 为一维 nFlat+1
    int64_t nFlat = mask_block_cnt.size(0) * mask_block_cnt.size(1) * mask_block_cnt.size(2);
    auto mask_offset = at::empty({nFlat + 1}, mask_block_cnt.options());
    auto full_offset = at::empty({nFlat + 1}, full_block_cnt.options());

    // compact 为一维 nFlat * maxBlocks
    int64_t maxBlocks = block_idx.size(3);
    auto mask_compact = at::empty({nFlat * maxBlocks}, mask_block_cnt.options());
    auto full_compact = at::empty({nFlat * maxBlocks}, block_idx.options());

    return {mask_cnt, mask_offset, mask_compact, full_cnt, full_offset, full_compact};
}

}  // anonymous namespace

// 注册 ln_mul 的 schema
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("compact_no_sync(Tensor mask_block_cnt, Tensor full_block_cnt, Tensor block_idx) -> Tensor[]");
}

// NPU 设备实现
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("compact_no_sync", &compact_no_sync_impl_npu);
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("compact_no_sync", &compact_no_sync_meta);
}
