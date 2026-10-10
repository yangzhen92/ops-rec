"""Guide8 regression tests for the unified host-side attention plan."""

import pytest
import torch
import torch_npu  # noqa: F401
import ops_rec  # noqa: F401
from ops_rec.ascendc.attention.hstu_v2 import hstu_v2

from backend.ascend_fuse_backend import Kernel


def _inputs(q_len=256, k_len=1024, heads=1, dim_qk=128, dim_v=64):
    torch.manual_seed(2026092608)
    q = torch.randn(q_len, heads, dim_qk, dtype=torch.float16)
    k = torch.randn(k_len, heads, dim_qk, dtype=torch.float16)
    v = torch.randn(k_len, heads, dim_v, dtype=torch.float16)
    return q, k, v


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_no_mask_plan_uses_shape_aware_tile_without_sparse(direction):
    q, k, v = _inputs()
    kernel = Kernel(0.5, 1 / 256, False, 256, 1024, [0, 256], [0, 1024])
    fwd_tile, bwd_tile, sparse, metadata, use_aicpu = kernel.create_plan(direction, q, k, v)
    selected = fwd_tile if direction == "forward" else bwd_tile
    assert selected[0] in (32, 64, 128, 256, 512)
    assert selected[1] in (32, 64, 128, 256, 512)
    assert selected[0] <= selected[1] if direction == "forward" else selected[0] >= selected[1]
    assert sparse == []
    assert metadata is None
    assert use_aicpu is False


def test_split_forward_and_backward_plan_interfaces():
    q, k, v = _inputs()
    offsets_q = torch.tensor([0, 256], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, 1024], dtype=torch.int32, device="npu")
    fwd_tile, fwd_sparse, fwd_metadata, fwd_aicpu = torch.ops.ops_rec.get_fwd_attn_plan(
        q.npu(), k.npu(), v.npu(), None, offsets_q, offsets_k, 256, 1024
    )
    bwd_tile, bwd_sparse, bwd_metadata, bwd_aicpu = torch.ops.ops_rec.get_bwd_attn_plan(
        q.npu(), k.npu(), v.npu(), None, offsets_q, offsets_k, 256, 1024
    )
    assert fwd_tile == [32, 128, 128]
    assert bwd_tile == [256, 64, 128]
    assert fwd_sparse == [] and bwd_sparse == []
    assert fwd_metadata is None and bwd_metadata is None
    assert not fwd_aicpu and not bwd_aicpu


def test_long_inner_sequence_keeps_aicore_schedule_without_measured_crossover():
    q, k, v = _inputs(q_len=128, k_len=8192, dim_qk=64, dim_v=32)
    kernel = Kernel(0.5, 1 / 128, False, 128, 8192, [0, 128], [0, 8192])
    tile, _, sparse, metadata, use_aicpu = kernel.create_plan("forward", q, k, v)
    torch.npu.synchronize()
    assert tile[0] == 32 and tile[1] == 256
    assert sparse == []
    assert metadata is None and use_aicpu is False
    without_metadata = kernel.forward(q, k, v, None, None, metadata, sparse or None)
    assert without_metadata.shape == (128, 1, 32)


def test_backward_metadata_uses_k_as_outer_axis():
    q_len, k_len, heads, dim = 7040, 256, 8, 256
    offsets_q = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, k_len], dtype=torch.int32, device="npu")
    metadata = torch.ops.ops_rec.create_attn_metadata(
        offsets_q,
        offsets_k,
        None,
        None,
        1,
        q_len,
        k_len,
        heads,
        heads,
        dim,
        0,
        -1,
        -1,
        "TND",
        "TND",
        "TND",
        None,
        None,
        1,
        1,
    )
    header = metadata[:16].cpu()
    records = metadata[16 : 16 + int(header[14]) * 16].reshape(-1, 16).cpu()
    # The final [bn2, m) endpoint is eight heads times four K-row tiles.
    assert int(header[2]) == 64
    assert (int(records[-1, 4]), int(records[-1, 3])) == (0, heads)


def test_direction_specific_l2_route_at_runtime_boundary():
    q_len, k_len, heads, dim = 256, 11520, 8, 256
    q = torch.empty(q_len, heads, dim, dtype=torch.float16, device="npu")
    k = torch.empty(k_len, heads, dim, dtype=torch.float16, device="npu")
    v = torch.empty_like(k)
    offsets_q = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, k_len], dtype=torch.int32, device="npu")
    fwd = torch.ops.ops_rec.get_fwd_attn_plan(q, k, v, None, offsets_q, offsets_k, q_len, k_len)
    bwd = torch.ops.ops_rec.get_bwd_attn_plan(k, q, q, None, offsets_k, offsets_q, k_len, q_len)
    torch.npu.synchronize()
    assert fwd[3] and fwd[2] is not None
    assert not bwd[3] and bwd[2] is None


@pytest.mark.parametrize("q_len,k_len,expect_aicpu", [(64, 128, False), (256, 11520, True)])
def test_forward_plan_aicore_and_aicpu_routes_match_direct_kernel(q_len, k_len, expect_aicpu):
    heads, dim = 8, 256
    torch.manual_seed(2026092702 + k_len)
    q = torch.empty(q_len, heads, dim, dtype=torch.float16, device="npu").uniform_(-0.1, 0.1)
    k = torch.empty(k_len, heads, dim, dtype=torch.float16, device="npu").uniform_(-0.1, 0.1)
    v = torch.empty_like(k).uniform_(-0.1, 0.1)
    oq = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
    ok = torch.tensor([0, k_len], dtype=torch.int32, device="npu")
    _, sparse, metadata, use_aicpu = torch.ops.ops_rec.get_fwd_attn_plan(
        q, k, v, None, oq, ok, q_len, k_len, None, None, 1, -1, 0, 0, True
    )
    assert use_aicpu is expect_aicpu
    planned = torch.ops.ops_rec.hstu_forward_v2(
        q,
        k,
        v,
        None,
        None,
        0,
        q_len,
        k_len,
        1 / q_len,
        oq,
        ok,
        None,
        None,
        1,
        0.5,
        metadata,
        sparse or None,
        -1,
        0,
        None,
        True,
    )
    direct = torch.ops.ops_rec.hstu_forward_v2(
        q, k, v, None, None, 0, q_len, k_len, 1 / q_len, oq, ok, None, None, 1, 0.5, None, None, -1, 0, None, True
    )
    torch.npu.synchronize()
    assert torch.allclose(planned.cpu(), direct.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128, 256])
def test_forward_stream_k_matches_deterministic(dtype, dim):
    torch.manual_seed(20261200 + dim + (1 if dtype == torch.bfloat16 else 0))
    q_len, k_len = 128, 3584
    q = torch.empty(q_len, 1, dim, dtype=dtype, device="npu").uniform_(-0.1, 0.1)
    k = torch.empty(k_len, 1, dim, dtype=dtype, device="npu").uniform_(-0.1, 0.1)
    v = torch.empty_like(k).uniform_(-0.1, 0.1)
    oq = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
    ok = torch.tensor([0, k_len], dtype=torch.int32, device="npu")
    reference = hstu_v2(q, k, v, None, 2, q_len, k_len, 1 / q_len, oq, ok, alpha=0.5, deterministic=True)
    actual = hstu_v2(q, k, v, None, 2, q_len, k_len, 1 / q_len, oq, ok, alpha=0.5, deterministic=False)
    torch.npu.synchronize()
    assert torch.allclose(actual.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)


def test_arbitrary_plan_matches_manual_packing_and_keeps_nine_tensor_abi():
    q, k, v = _inputs(q_len=256, k_len=1024)
    offsets_q = torch.tensor([0, 256], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, 1024], dtype=torch.int32, device="npu")
    context = torch.tensor([8], dtype=torch.int32, device="npu")
    target = torch.tensor([32], dtype=torch.int32, device="npu")
    af = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(offsets_k, context, target, 4, 1024)
    kernel = Kernel(0.5, 1 / 256, False, 256, 1024, [0, 256], [0, 1024])
    tile, _, sparse, metadata, use_aicpu = kernel.create_plan("forward", q, k, v, af.cpu(), context, target, 4)
    manual = torch.ops.ops_rec.create_fwd_packed_block_sparse(af, offsets_q, offsets_k, 256, 1024, tile[0], tile[1])
    torch.npu.synchronize()
    assert tile == [32, 128, 128]
    assert len(sparse) == 9 and len(manual) == 9
    assert sparse[2].numel() == 1
    for index in (0, 3):
        assert torch.equal(sparse[index].cpu(), manual[index].cpu())
    for actual, expected in zip(sparse, manual):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for index in (2, 4, 5, 7):
        assert sparse[index].shape == torch.Size([1])
    planned_output = kernel.forward(q, k, v, None, None, metadata, sparse, context, target, 4)
    manual_output = kernel.forward(q, k, v, None, None, None, list(manual), context, target, 4)
    assert torch.equal(planned_output, manual_output)
    assert metadata is None and use_aicpu is False


def test_split_attn_plan_fake_tensor_meta_path():
    with torch._subclasses.fake_tensor.FakeTensorMode():
        q = torch.empty(256, 1, 128, device="npu", dtype=torch.float16)
        k = torch.empty(1024, 1, 128, device="npu", dtype=torch.float16)
        v = torch.empty(1024, 1, 64, device="npu", dtype=torch.float16)
        offsets_q = torch.empty(2, device="npu", dtype=torch.int32)
        offsets_k = torch.empty(2, device="npu", dtype=torch.int32)
        fwd = torch.ops.ops_rec.get_fwd_attn_plan(q, k, v, None, offsets_q, offsets_k, 256, 1024)
        bwd = torch.ops.ops_rec.get_bwd_attn_plan(q, k, v, None, offsets_q, offsets_k, 256, 1024)
        assert fwd[0] == [128, 128, 128] and bwd[0] == [256, 128, 128]
        assert fwd[1] == [] and bwd[1] == [] and fwd[2] is None and bwd[2] is None


def test_split_plan_fake_tensor_masked_structure():
    with torch._subclasses.fake_tensor.FakeTensorMode():
        q = torch.empty(64, 1, 32, device="npu", dtype=torch.float16)
        k = torch.empty(128, 1, 32, device="npu", dtype=torch.float16)
        offsets_q = torch.empty(2, device="npu", dtype=torch.int32)
        offsets_k = torch.empty(2, device="npu", dtype=torch.int32)
        af = torch.empty(128, 8, device="npu", dtype=torch.int32)
        fwd = torch.ops.ops_rec.get_fwd_attn_plan(q, k, k, af, offsets_q, offsets_k, 64, 128)
        bwd = torch.ops.ops_rec.get_bwd_attn_plan(q, k, k, af, offsets_q, offsets_k, 64, 128)
        assert len(fwd[1]) == 9 and len(bwd[1]) == 9
        assert fwd[2] is None and bwd[2] is None
        assert not fwd[3] and not bwd[3]


def test_unified_plan_autograd_matches_direct_backward():
    q, k, v = (tensor.npu().requires_grad_(True) for tensor in _inputs(q_len=64, k_len=128, dim_qk=32, dim_v=32))
    grad = torch.randn(64, 1, 32, dtype=torch.float16, device="npu")
    offsets_q = torch.tensor([0, 64], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, 128], dtype=torch.int32, device="npu")
    out = hstu_v2(q, k, v, None, 2, 64, 128, 1 / 64, offsets_q, offsets_k, alpha=0.5)
    out.backward(grad)
    expected = torch.ops.ops_rec.hstu_backward_v2(
        grad,
        q.detach(),
        k.detach(),
        v.detach(),
        64,
        128,
        offsets_q,
        offsets_k,
        None,
        None,
        None,
        1 / 64,
        1,
        0.5,
        -1,
        -1,
        None,
        None,
    )
    torch.npu.synchronize()
    actual_grads = (q.grad, k.grad, v.grad)
    # dQ is accumulated atomically across K tiles; repeated launches may
    # differ by one fp16 ULP solely from legal atomic ordering. dK/dV have a
    # single owner and remain bitwise deterministic.
    assert torch.allclose(actual_grads[0].cpu(), expected[0].cpu(), rtol=1e-3, atol=1e-3)
    for actual, reference in zip(actual_grads[1:], expected[1:3]):
        assert torch.equal(actual.cpu(), reference.cpu())


def test_unified_plan_is_captured_with_hstu_in_full_graph():
    q, k, v = (tensor.npu() for tensor in _inputs(q_len=64, k_len=128, dim_qk=32, dim_v=32))
    offsets_q = torch.tensor([0, 64], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, 128], dtype=torch.int32, device="npu")
    captured = []

    def backend(graph_module, _example_inputs):
        captured.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(
        lambda q_, k_, v_: hstu_v2(q_, k_, v_, None, 2, 64, 128, 1 / 64, offsets_q, offsets_k, alpha=0.5),
        backend=backend,
        fullgraph=True,
    )
    expected = hstu_v2(q, k, v, None, 2, 64, 128, 1 / 64, offsets_q, offsets_k, alpha=0.5)
    actual = compiled(q, k, v)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected.cpu())
    graph = "\n".join(str(module.graph) for module in captured)
    assert "ops_rec.get_fwd_attn_plan" in graph
    assert "ops_rec.hstu_forward_v2" in graph


def test_compiled_forward_runs_lazy_backward_plan():
    q, k, v = (tensor.npu().requires_grad_(True) for tensor in _inputs(64, 128, 1, 32, 32))
    grad = torch.randn(64, 1, 32, dtype=torch.float16, device="npu")
    offsets_q = torch.tensor([0, 64], dtype=torch.int32, device="npu")
    offsets_k = torch.tensor([0, 128], dtype=torch.int32, device="npu")
    compiled = torch.compile(
        lambda q_, k_, v_: hstu_v2(q_, k_, v_, None, 2, 64, 128, 1 / 64, offsets_q, offsets_k, alpha=0.5),
        backend="eager",
        fullgraph=True,
    )
    compiled(q, k, v).backward(grad)
    reference = torch.ops.ops_rec.hstu_backward_v2(
        grad,
        q.detach(),
        k.detach(),
        v.detach(),
        64,
        128,
        offsets_q,
        offsets_k,
        None,
        None,
        None,
        1 / 64,
        1,
        0.5,
        -1,
        -1,
        None,
        None,
    )
    torch.npu.synchronize()
    assert torch.allclose(q.grad.cpu(), reference[0].cpu(), rtol=1e-3, atol=1e-3)
    assert torch.equal(k.grad.cpu(), reference[1].cpu())
    assert torch.equal(v.grad.cpu(), reference[2].cpu())


def test_unequal_batch_lengths_and_qkv_dimensions_use_one_plan():
    torch.manual_seed(2026092618)
    q_lens, k_lens = (65, 127), (193, 321)
    q_offsets = [0, q_lens[0], sum(q_lens)]
    k_offsets = [0, k_lens[0], sum(k_lens)]
    q = torch.randn(sum(q_lens), 2, 64, dtype=torch.float16, device="npu")
    k = torch.randn(sum(k_lens), 2, 64, dtype=torch.float16, device="npu")
    v = torch.randn(sum(k_lens), 2, 32, dtype=torch.float16, device="npu")
    oq = torch.tensor(q_offsets, dtype=torch.int32, device="npu")
    ok = torch.tensor(k_offsets, dtype=torch.int32, device="npu")
    context = torch.tensor([0, 8], dtype=torch.int32, device="npu")
    target = torch.tensor([16, 32], dtype=torch.int32, device="npu")
    af = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(ok, context, target, 4, sum(k_lens))
    tile, sparse, metadata, use_aicpu = torch.ops.ops_rec.get_fwd_attn_plan(
        q, k, v, af, oq, ok, max(q_lens), max(k_lens), context, target, 4, -1, 0, 0
    )
    output = torch.ops.ops_rec.hstu_forward_v2(
        q,
        k,
        v,
        None,
        None,
        0,
        max(q_lens),
        max(k_lens),
        1 / max(q_lens),
        oq,
        ok,
        context,
        target,
        4,
        0.5,
        metadata,
        sparse,
        -1,
        0,
    )
    torch.npu.synchronize()
    assert tile == [32, 256, 64]
    assert len(sparse) == 9 and not use_aicpu and metadata is None
    assert output.shape == (sum(q_lens), 2, 32)
