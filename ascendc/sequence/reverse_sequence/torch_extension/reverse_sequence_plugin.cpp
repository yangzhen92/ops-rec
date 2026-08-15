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

// 声明本插件依赖的 vendor（aclnn 接口所在 op_api 库），必须在 include "pytorch_npu_helper.hpp" 之前定义
#define OPS_REC_CUST_VENDORS "reverse_sequence"

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

namespace npu_reverse_sequence {

constexpr int64_t INPUT_DIM = 3;
constexpr int64_t SEQ_LENGTHS_DIM = 1;
constexpr int64_t EMB_DIM_MIN = 1;
constexpr int64_t EMB_DIM_MAX = 1024;

constexpr int64_t MAX_SEQ_LENGTH_DIM_INDEX = 1;
constexpr int64_t EMB_DIM_INDEX = 2;

constexpr int64_t BATCH_SIZE_MAX = 10240;
constexpr int64_t MAX_SEQ_LENGTH_MAX = 102400;

// reverse 是对合运算（再做一次即还原），反向等价于对 grad_output 再执行一次 reverse
class ReverseSequenceOp : public torch::autograd::Function<ReverseSequenceOp> {
public:
    static at::Tensor forward(AutogradContext* ctx, const at::Tensor& input, const at::Tensor& seq_lengths)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({seq_lengths});

        // aclnnReverseSequenceCust 仅将每个 batch 前 seq_len 行逆序写入 output，
        // 其余行不会被 kernel 触碰，因此 output 必须初始化为 input 的拷贝而非空 buffer。
        auto output = input.clone();
        EXEC_NPU_CMD(aclnnReverseSequenceCust, input, seq_lengths, output);
        return output;
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto seq_lengths = saved[0];
        check_tensor_non_empty(seq_lengths, "seq_lengths");
        auto grad_output_conti = grad_output.contiguous();
        auto output = grad_output_conti.clone();
        EXEC_NPU_CMD(aclnnReverseSequenceCust, grad_output_conti, seq_lengths, output);
        // 反向在 autograd 引擎线程上执行，若异步 kernel 仍在运行而主线程已释放/复用其输入显存，
        // kernel 会读到野数据导致 vector core 超时（507034）；此处同步等待 kernel 完成。
        c10_npu::getCurrentNPUStream().synchronize();
        return {output, Variable()};
    }
};

at::Tensor npu_reverse_sequence_impl(const at::Tensor& input, const at::Tensor& seq_lengths)
{
    check_tensor_non_empty(input, "input");
    check_tensor_non_empty(seq_lengths, "seq_lengths");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {input, seq_lengths};
    std::vector<std::string> names = {"input", "seq_lengths"};
    check_tensor_npu_device(tensors, names);

    check_tensor_dim(input, INPUT_DIM, "reverse_sequence input");
    check_tensor_dim(seq_lengths, SEQ_LENGTHS_DIM, "reverse_sequence seq_lengths");
    TORCH_CHECK(input.size(0) == seq_lengths.size(0), "param input dim 0: ", input.size(0),
                " is not equal to seq_lengths dim 0: ", seq_lengths.size(0));
    TORCH_CHECK(input.size(0) <= BATCH_SIZE_MAX, "input dim[0] is batch_size, must be less or equal to 10240, but got ",
                input.size(0));
    TORCH_CHECK(input.size(MAX_SEQ_LENGTH_DIM_INDEX) <= MAX_SEQ_LENGTH_MAX,
                "input dim[1] is max_seq_length, must be less or equal to 102400, but got ",
                input.size(MAX_SEQ_LENGTH_DIM_INDEX));
    int64_t emb_dim = input.size(EMB_DIM_INDEX);
    TORCH_CHECK(emb_dim >= EMB_DIM_MIN && emb_dim <= EMB_DIM_MAX, "emb_dim must be in range:[", EMB_DIM_MIN, ", ",
                EMB_DIM_MAX, "], but got ", emb_dim);
    auto input_conti = input.contiguous();
    auto seq_lengths_conti = seq_lengths.contiguous();
    return ReverseSequenceOp::apply(input_conti, seq_lengths_conti);
}
}  // namespace npu_reverse_sequence

// 注册 reverse_sequence 的 schema
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("reverse_sequence(Tensor input, Tensor seq_lengths) -> Tensor");
}

// NPU 设备实现（PrivateUse1 = Ascend NPU 后端标识）。
// 注意：不要给本算子注册真实的 Autograd kernel —— 它会在 dynamo fake tensor 传播阶段拦截
// 调用并执行真实 aclnn 路径，导致 torch.compile 失败；反向由 impl 内部的
// autograd.Function（apply）自行建立。
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("reverse_sequence", &npu_reverse_sequence::npu_reverse_sequence_impl);
}

// 显式 Autograd fallthrough：反向由 impl 内部的 autograd.Function 建立，
// 避免 dispatcher 对"未注册 Autograd kernel 却参与反向传播"的弃用告警
TORCH_LIBRARY_IMPL(ops_rec, Autograd, m)
{
    m.impl("reverse_sequence", torch::CppFunction::makeFallthrough());
}

namespace {

at::Tensor reverse_sequence_meta(const at::Tensor& input, const at::Tensor& seq_lengths)
{
    check_tensor_dim(input, npu_reverse_sequence::INPUT_DIM, "reverse_sequence input");
    check_tensor_dim(seq_lengths, npu_reverse_sequence::SEQ_LENGTHS_DIM, "reverse_sequence seq_lengths");
    return at::empty_like(input, input.options());
}

}  // anonymous namespace

// Meta 实现，用于 torch.compile 的 shape 推断
TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("reverse_sequence", &reverse_sequence_meta);
}
