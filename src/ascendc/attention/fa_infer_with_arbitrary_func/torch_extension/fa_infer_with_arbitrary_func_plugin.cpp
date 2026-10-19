/**
 * @file fa_infer_with_arbitrary_func_plugin.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#define OPS_REC_CUST_VENDORS "fa_infer_with_arbitrary_func_forward"
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
#include "common_utils.h"
using tensor_list = std::vector<at::Tensor>;
using namespace at;

namespace {
constexpr int EXPECTED_DIM_3D = 3;
constexpr int EXPECTED_DIM_4D = 4;
constexpr int64_t EXPECTED_HEAD_DIM = 128;
}  // namespace

Tensor fa_infer_with_arbitrary_func_impl_npu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                             const at::Tensor& arbitrary_func, const tensor_list& q2k_block_sparse)
{
    auto out = at::empty_like(q);
    TORCH_CHECK(q2k_block_sparse.size() == 6, "q2k_block_sparse size must be 6 tensors");
    // TND格式
    check_tensor_dim(q, EXPECTED_DIM_3D, "q");
    check_tensor_dim(k, EXPECTED_DIM_3D, "k");
    check_tensor_dim(v, EXPECTED_DIM_3D, "v");
    TORCH_CHECK(q.size(-1) == EXPECTED_HEAD_DIM, "q last dim must be ", EXPECTED_HEAD_DIM, ", but got ", q.size(-1));
    TORCH_CHECK(k.size(-1) == EXPECTED_HEAD_DIM, "k last dim must be ", EXPECTED_HEAD_DIM, ", but got ", k.size(-1));
    TORCH_CHECK(v.size(-1) == EXPECTED_HEAD_DIM, "v last dim must be ", EXPECTED_HEAD_DIM, ", but got ", v.size(-1));
    check_tensor_dim(arbitrary_func, EXPECTED_DIM_4D, "arbitrary_func");
    TORCH_CHECK(arbitrary_func.size(0) == 1 && arbitrary_func.size(1) == 1 && arbitrary_func.size(2) >= q.size(0) &&
                    arbitrary_func.size(3) == 4,
                "arbitrary_func first two dim must be 1, and third dim must be >= q.size(0), and last dim must be 4, "
                "but got ",
                arbitrary_func.size(0), " and ", arbitrary_func.size(1), " and ", arbitrary_func.size(2), " and ",
                arbitrary_func.size(3));

    // arbitrary FA 不使用 mask / paged-attn block_tables，生成空Tensor传给aclnn
    at::Tensor mask = at::Tensor();
    at::Tensor block_tables = at::Tensor();
    auto mask_block_cnt = q2k_block_sparse[0].contiguous();
    auto mask_block_offset = q2k_block_sparse[1].contiguous();
    auto mask_block_idx = q2k_block_sparse[2].contiguous();
    auto full_block_cnt = q2k_block_sparse[3].contiguous();
    auto fullblock_offset = q2k_block_sparse[4].contiguous();
    auto fullblock_idx = q2k_block_sparse[5].contiguous();
    auto arbitrary_func_conti = arbitrary_func.contiguous();
    EXEC_NPU_CMD(aclnnFaInferWithArbitraryFunc, q, k, v, mask, block_tables, arbitrary_func_conti, mask_block_cnt,
                 mask_block_offset, mask_block_idx, full_block_cnt, fullblock_offset, fullblock_idx, out);
    return out;
}

namespace {
at::Tensor fa_infer_with_arbitrary_func_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                             const at::Tensor& arbitrary_func, const tensor_list& q2k_block_sparse)
{
    return at::empty_like(q);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("fa_infer_with_arbitrary_func(Tensor q, "
          "                             Tensor k, "
          "                             Tensor v, "
          "                             Tensor arbitrary_func, "
          "                             Tensor[] q2k_block_sparse) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("fa_infer_with_arbitrary_func", &fa_infer_with_arbitrary_func_impl_npu);
}

TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("fa_infer_with_arbitrary_func", TORCH_FN(fa_infer_with_arbitrary_func_meta));
}
