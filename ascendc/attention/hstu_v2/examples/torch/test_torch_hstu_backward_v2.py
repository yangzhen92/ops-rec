"""
Torch 调用 hstu_backward_v2 示例

运行前需安装 ops_rec 或配置环境:
    python test_torch_hstu_backward_v2.py
"""

import torch
import ops_rec


def test_hstu_backward_v2_basic():
    torch.npu.set_device(0)
    batch_size = 2
    heads = 4
    dim_qk = 64
    dim_gv = 64
    max_seqlen_q = 128
    max_seqlen_k = 128
    total_seq_len_q = 200
    total_seq_len_k = 200

    grad = torch.randn(total_seq_len_q, heads, dim_gv, dtype=torch.float16, device="npu:0")
    q = torch.randn(total_seq_len_q, heads, dim_qk, dtype=torch.float16, device="npu:0")
    k = torch.randn(total_seq_len_k, heads, dim_qk, dtype=torch.float16, device="npu:0")
    v = torch.randn(total_seq_len_k, heads, dim_gv, dtype=torch.float16, device="npu:0")

    seq_lens_q = torch.tensor([100, 100], dtype=torch.int32)
    seq_lens_k = torch.tensor([100, 100], dtype=torch.int32)
    seq_offset_q = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_q, dim=0)])
    seq_offset_k = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_k, dim=0)])

    rab = torch.rand(batch_size, heads, max_seqlen_q, max_seqlen_k, dtype=torch.float16).uniform_(-1, 1)

    # causal mask example
    q_grad, k_grad, v_grad, rab_grad = ops_rec.ascendc.attention.hstu_backward_v2(
        grad,
        q,
        k,
        v,
        max_seqlen_q,
        max_seqlen_k,
        seq_offset_q.to("npu"),
        seq_offset_k.to("npu"),
        rab,
        None,
        None,
        0.0,
        1,
        1.0,
        -1,
        0,
    )

    print(f"q_grad: {q_grad.shape}")
    print(f"k_grad: {k_grad.shape}")
    print(f"v_grad: {v_grad.shape}")
    print(f"rab_grad: {rab_grad.shape}")
    print("PASSED")


if __name__ == "__main__":
    test_hstu_backward_v2_basic()
