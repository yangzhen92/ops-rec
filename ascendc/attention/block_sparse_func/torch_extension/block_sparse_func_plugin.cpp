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
#define OPS_REC_CUST_VENDORS "block_sparse_func"

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

namespace {
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
}  // namespace

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

// ---------------------------------------------------------------------------
// create_q2k_block_sparse_from_func：由 func_tensor 生成 Q2K 块稀疏张量
// ---------------------------------------------------------------------------
void validate_create_q2k_inputs(const Tensor& func_tensor)
{
    // 检查设备
    check_tensor_npu_device({func_tensor}, {"func_tensor"});
    // 检查数据类型
    TORCH_CHECK(func_tensor.dtype() == at::kInt, "func_tensor must be int32");
    // 检查维度 [B, H, func_q_len, n_func]
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D [B, H, func_q_len, n_func]");
}

/**
 * create_q2k_block_sparse_from_func算子的NPU实现
 * @param func_tensor [B, H, func_q_len, n_func] 函数编码张量
 * @param q_len Q 序列长度
 * @param kv_len KV 序列长度
 * @param q_block_size Q 块大小
 * @param kv_block_size KV 块大小
 * @param check_q_boundary 部分 q_block 是否不允许 FULL kv_block
 * @return 元组 (mask_block_cnt, full_block_cnt, block_idx)
 */
std::tuple<Tensor, Tensor, Tensor> create_q2k_block_sparse_from_func_impl_npu(const Tensor& func_tensor, int64_t q_len,
                                                                              int64_t kv_len, int64_t q_block_size,
                                                                              int64_t kv_block_size,
                                                                              bool check_q_boundary, bool debug)
{
    validate_create_q2k_inputs(func_tensor);

    TORCH_CHECK(q_len > 0, "q_len must be positive");
    TORCH_CHECK(kv_len > 0, "kv_len must be positive");
    // catlass 限制
    TORCH_CHECK(q_block_size > 0 && q_block_size % 32 == 0, "q_block_size must be positive and divisible by 32");
    TORCH_CHECK(kv_block_size > 0 && kv_block_size % 32 == 0, "kv_block_size must be positive and divisible by 32");

    // func_tensor 可为非连续张量，保留 strides 以便 kernel 按步长索引
    int32_t B = func_tensor.size(0);
    int32_t H = func_tensor.size(1);
    int32_t func_q_len = func_tensor.size(2);
    int32_t n_func = func_tensor.size(3);
    TORCH_CHECK(func_q_len >= q_len, "func_q_len must be greater than or equal to q_len");
    TORCH_CHECK(n_func % 2 == 0, "n_func must be even (e.g., 2, 4, 6, 8, ...)");

    int32_t stride_b = func_tensor.stride(0);
    int32_t stride_h = func_tensor.stride(1);
    int32_t stride_q = func_tensor.stride(2);
    int32_t stride_f = func_tensor.stride(3);

    int32_t num_q_blocks = DIVUP(q_len, q_block_size);
    int32_t num_kv_blocks = DIVUP(kv_len, kv_block_size);

    // 分配输出张量
    Tensor mask_block_cnt = debug ? at::full({B, H, num_q_blocks}, -1, func_tensor.options())
                                  : at::empty({B, H, num_q_blocks}, func_tensor.options());
    Tensor full_block_cnt = debug ? at::full({B, H, num_q_blocks}, -1, func_tensor.options())
                                  : at::empty({B, H, num_q_blocks}, func_tensor.options());
    Tensor block_idx = debug ? at::full({B, H, num_q_blocks, num_kv_blocks}, -1, func_tensor.options())
                             : at::empty({B, H, num_q_blocks, num_kv_blocks}, func_tensor.options());

    // 调用 create_q2k_block_sparse_from_func kernel
    EXEC_NPU_CMD(aclnnCreateQ2kBlockSparseFromFunc, func_tensor, stride_b, stride_h, stride_q, stride_f, q_len, kv_len,
                 q_block_size, kv_block_size, check_q_boundary, mask_block_cnt, full_block_cnt, block_idx);

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

namespace {
/**
 * create_q2k_block_sparse_from_func算子的元函数实现
 */
std::tuple<Tensor, Tensor, Tensor> create_q2k_block_sparse_from_func_meta(const Tensor& func_tensor, int64_t q_len,
                                                                          int64_t kv_len, int64_t q_block_size,
                                                                          int64_t kv_block_size, bool check_q_boundary,
                                                                          bool debug)
{
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D");
    int32_t B = func_tensor.size(0);
    int32_t H = func_tensor.size(1);
    int32_t num_q_blocks = DIVUP(q_len, q_block_size);
    int32_t num_kv_blocks = DIVUP(kv_len, kv_block_size);

    auto mask_block_cnt = at::empty({B, H, num_q_blocks}, func_tensor.options());
    auto full_block_cnt = at::empty({B, H, num_q_blocks}, func_tensor.options());
    auto block_idx = at::empty({B, H, num_q_blocks, num_kv_blocks}, func_tensor.options());

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// create_k2q_block_sparse_from_func：由 func_tensor 生成 K2Q 块稀疏张量（反向）
// ---------------------------------------------------------------------------
/**
 * create_k2q_block_sparse_from_func算子的NPU实现（反向/transpose）
 * @param func_tensor [B, H, func_q_len, n_func] 函数编码张量
 * @param q_len Q 序列长度
 * @param kv_len KV 序列长度
 * @param q_block_size Q 块大小
 * @param kv_block_size KV 块大小
 * @return 元组 (mask_block_cnt, full_block_cnt, block_idx)
 *
 * K2Q 与 Q2K 共享 func_tensor 布局，但输出形状不同：
 *   mask_block_cnt/full_block_cnt: [B, H, numKvBlocks]
 *   block_idx: [B, H, numKvBlocks, numQBlocks]
 * K2Q 始终进行 q_boundary 检查（部分 q_block 不允许 FULL），故无 check_q_boundary 参数。
 */
std::tuple<Tensor, Tensor, Tensor> create_k2q_block_sparse_from_func_impl_npu(const Tensor& func_tensor, int64_t q_len,
                                                                              int64_t kv_len, int64_t q_block_size,
                                                                              int64_t kv_block_size, bool debug)
{
    validate_create_q2k_inputs(func_tensor);

    TORCH_CHECK(q_len > 0, "q_len must be positive");
    TORCH_CHECK(kv_len > 0, "kv_len must be positive");
    TORCH_CHECK(q_block_size > 0 && q_block_size % 32 == 0, "q_block_size must be positive and divisible by 32");
    TORCH_CHECK(kv_block_size > 0 && kv_block_size % 32 == 0, "kv_block_size must be positive and divisible by 32");

    int32_t B = func_tensor.size(0);
    int32_t H = func_tensor.size(1);
    int32_t func_q_len = func_tensor.size(2);
    int32_t n_func = func_tensor.size(3);
    TORCH_CHECK(func_q_len >= q_len, "func_q_len must be greater than or equal to q_len");
    TORCH_CHECK(n_func % 2 == 0, "n_func must be even (e.g., 2, 4, 6, 8, ...)");

    // func_tensor 可为非连续张量，保留 strides 以便 kernel 按步长索引
    int32_t stride_b = func_tensor.stride(0);
    int32_t stride_h = func_tensor.stride(1);
    int32_t stride_q = func_tensor.stride(2);
    int32_t stride_f = func_tensor.stride(3);

    int32_t num_q_blocks = DIVUP(q_len, q_block_size);
    int32_t num_kv_blocks = DIVUP(kv_len, kv_block_size);
    // 分配输出张量
    Tensor mask_block_cnt = debug ? at::full({B, H, num_kv_blocks}, -1, func_tensor.options())
                                  : at::empty({B, H, num_kv_blocks}, func_tensor.options());
    Tensor full_block_cnt = debug ? at::full({B, H, num_kv_blocks}, -1, func_tensor.options())
                                  : at::empty({B, H, num_kv_blocks}, func_tensor.options());
    Tensor block_idx = debug ? at::full({B, H, num_kv_blocks, num_q_blocks}, -1, func_tensor.options())
                             : at::empty({B, H, num_kv_blocks, num_q_blocks}, func_tensor.options());

    // K2Q 前置阶段：预计算每个 (b, h, q_block) 的 q_block_kv_min / q_block_kv_max，
    // 供 create_k2q_block_sparse_from_func 跳过 kv 范围无交叠的 q_block（kv-range 优化）。
    Tensor q_block_kv_min = at::empty({B, H, num_q_blocks}, func_tensor.options());
    Tensor q_block_kv_max = at::empty({B, H, num_q_blocks}, func_tensor.options());
    EXEC_NPU_CMD(aclnnComputeQBlockKvRange, func_tensor, stride_b, stride_h, stride_q, stride_f, q_len, q_block_size,
                 q_block_kv_min, q_block_kv_max);

    // 调用 create_k2q_block_sparse_from_func kernel
    EXEC_NPU_CMD(aclnnCreateK2qBlockSparseFromFunc, func_tensor, q_block_kv_min, q_block_kv_max, stride_b, stride_h,
                 stride_q, stride_f, q_len, kv_len, q_block_size, kv_block_size, mask_block_cnt, full_block_cnt,
                 block_idx);

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

namespace {
/**
 * create_k2q_block_sparse_from_func算子的元函数实现
 */
std::tuple<Tensor, Tensor, Tensor> create_k2q_block_sparse_from_func_meta(const Tensor& func_tensor, int64_t q_len,
                                                                          int64_t kv_len, int64_t q_block_size,
                                                                          int64_t kv_block_size, bool debug)
{
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D");
    int64_t B = func_tensor.size(0);
    int64_t H = func_tensor.size(1);
    int64_t num_q_blocks = DIVUP(q_len, q_block_size);
    int64_t num_kv_blocks = DIVUP(kv_len, kv_block_size);

    auto mask_block_cnt = at::empty({B, H, num_kv_blocks}, func_tensor.options());
    auto full_block_cnt = at::empty({B, H, num_kv_blocks}, func_tensor.options());
    auto block_idx = at::empty({B, H, num_kv_blocks, num_q_blocks}, func_tensor.options());

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

}  // anonymous namespace

// 注册 block_sparse_func 的 schema
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("compact_no_sync(Tensor mask_block_cnt, Tensor full_block_cnt, Tensor block_idx) -> Tensor[]");
    m.def("create_q2k_block_sparse_from_func(Tensor func_tensor, int q_len, int kv_len, int q_block_size=128, "
          "int kv_block_size=128, bool check_q_boundary=False, bool debug=False) -> (Tensor, Tensor, Tensor)");
    m.def("create_k2q_block_sparse_from_func(Tensor func_tensor, int q_len, int kv_len, int q_block_size=128, "
          "int kv_block_size=128, bool debug=False) -> (Tensor, Tensor, Tensor)");
}

// NPU 设备实现
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("compact_no_sync", &compact_no_sync_impl_npu);
    m.impl("create_q2k_block_sparse_from_func", &create_q2k_block_sparse_from_func_impl_npu);
    m.impl("create_k2q_block_sparse_from_func", &create_k2q_block_sparse_from_func_impl_npu);
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("compact_no_sync", &compact_no_sync_meta);
    m.impl("create_q2k_block_sparse_from_func", &create_q2k_block_sparse_from_func_meta);
    m.impl("create_k2q_block_sparse_from_func", &create_k2q_block_sparse_from_func_meta);
}
