"""
Torch 调用 hstu_backward_v2 示例

运行前需安装 ops_rec 或配置环境:
    python test_torch_hstu_backward_v2.py
"""

import torch
import ops_rec

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False


def test_hstu_backward_v2_basic():
    batch_size = 2
    heads = 4
    dim_qk = 64
    dim_gv = 64
    max_seqlen_q = 128
    max_seqlen_k = 128
    total_seq_len_q = 200
    total_seq_len_k = 200
    num_context = None
    num_target = None
    target_group_size = 1
    silu_scale = 1 / max_seqlen_q
    alpha = 0.5

    grad = torch.rand(total_seq_len_q, heads, dim_gv, dtype=torch.float16).uniform_(-1, 1)
    q = torch.rand(total_seq_len_q, heads, dim_qk, dtype=torch.float16).uniform_(-1, 1)
    k = torch.rand(total_seq_len_k, heads, dim_qk, dtype=torch.float16).uniform_(-1, 1)
    v = torch.rand(total_seq_len_k, heads, dim_gv, dtype=torch.float16).uniform_(-1, 1)

    seq_lens_q = torch.tensor([100, 100], dtype=torch.int32)
    seq_lens_k = torch.tensor([100, 100], dtype=torch.int32)
    seq_offset_q = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_q, axis=0))).numpy()
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_k, axis=0))).numpy()
    seq_offset_q = torch.Tensor(seq_offset_q).to("npu").to(torch.int32)
    seq_offset_k = torch.Tensor(seq_offset_k).to("npu").to(torch.int32)

    rab = torch.rand(batch_size, heads, max_seqlen_q, max_seqlen_k, dtype=torch.float16).uniform_(-1, 1)

    grad = grad.to("npu")
    q = q.to("npu")
    k = k.to("npu")
    v = v.to("npu")

    if rab is not None:
        rab = rab.to("npu")
    else:
        rab = None

    # causal mask example
    q_grad, k_grad, v_grad, rab_grad = ops_rec.ascendc.attention.hstu_backward_v2(
        grad,
        q,
        k,
        v,
        max_seqlen_q,
        max_seqlen_k,
        seq_offset_q,
        seq_offset_k,
        rab,
        num_context,
        num_target,
        silu_scale,
        target_group_size,
        alpha,
        -1,
        -1,
    )
    torch.npu.synchronize()

    print(f"q_grad: {q_grad}")
    print(f"q_grad: {q_grad.shape}")
    print(f"k_grad: {k_grad.shape}")
    print(f"v_grad: {v_grad.shape}")
    print(f"rab_grad: {rab_grad.shape}")
    print("PASSED")


if __name__ == "__main__":
    test_hstu_backward_v2_basic()
