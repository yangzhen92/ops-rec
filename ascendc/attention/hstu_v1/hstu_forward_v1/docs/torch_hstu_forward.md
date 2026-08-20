# hstu_dense / hstu_jagged / hstu_paged

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

HSTU（Hierarchical Sequential Transduction Unit）融合注意力算子，支持 normal（dense）、jagged、paged 三种数据格式，支持 GQA（K/V 头数小于 Q 头数）与 qk/v dim 不等，自动微分（autograd）由 C++ 侧 autograd 包装提供。

计算公式：

```text
HSTU(q, k, v, mask, bias, siluScale) = (Silu(qk^T + bias) * siluScale * mask) v
```

## 接口原型

```python
ops_rec.ascendc.attention.hstu_dense(
    q: Tensor, k: Tensor, v: Tensor,
    mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None,
    mask_type: int = 0, max_seq_len: int = 0, silu_scale: float = 0.0) -> Tensor

ops_rec.ascendc.attention.hstu_jagged.equal(
    q: Tensor, k: Tensor, v: Tensor,
    mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None,
    mask_type: int = 0, max_seq_len: int = 0, silu_scale: float = 0.0,
    seq_offset: Tensor = None, num_context: Optional[Tensor] = None,
    num_target: Optional[Tensor] = None, target_group_size: Optional[int] = 0,
    alpha: Optional[float] = 1.0, deterministic: bool = False) -> Tensor

ops_rec.ascendc.attention.hstu_jagged.delta(
    q: Tensor, k: Tensor, v: Tensor,
    mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None,
    mask_type: int = 0, max_seq_len: int = 0, max_seq_len_k: int = 0,
    silu_scale: float = 0.0, seq_offset: Tensor = None, seq_offset_k: Tensor = None,
    num_context: Optional[Tensor] = None, num_target: Optional[Tensor] = None,
    target_group_size: Optional[int] = 0, alpha: Optional[float] = 1.0,
    deterministic: bool = False) -> Tensor

ops_rec.ascendc.attention.hstu_paged(
    q: Tensor, k: Tensor, v: Tensor, kv_cache: Tensor = None,
    mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None,
    mask_type: int = 0, max_seq_len: int = 0, max_seq_len_k: int = 0,
    silu_scale: float = 0.0, seq_offset: Tensor = None, seq_offset_k: Tensor = None,
    seq_offset_t: Tensor = None, page_offsets: Tensor = None, page_ids: Tensor = None,
    last_page_len: Tensor = None, num_target: Tensor = None,
    target_group_size: int = 0, alpha: Optional[float] = 1.0,
    deterministic: bool = False) -> Tensor
```

> 反向接口 `hstu_dense_backward` / `hstu_jagged_backward.equal|delta` 自动挂载在 autograd 图中，一般无需直接调用。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| q | 输入 | Tensor | FLOAT32/FLOAT16/BFLOAT16（A5 另支持 FP8_E4M3FN） | ND | [B, S, N_q, D_q] 或 [s_b, N_q, D_q] | Query 张量 |
| k | 输入 | Tensor | 同 q | ND | [B, S, N_k, D_q] 或 [s_b, N_k, D_q] | Key 张量（GQA 时 N_k < N_q） |
| v | 输入 | Tensor | 同 q | ND | [B, S, N_k, D_v] 或 [s_b, N_k, D_v] | Value 张量 |
| mask | 输入 | Tensor（可选） | 同 q | ND | [B, N, S, S] | mask_type=3 时必传 |
| attn_bias | 输入 | Tensor（可选） | 同 q | ND | [B, N, S, S] | 相对位置偏置 |
| seq_offset / seq_offset_k / seq_offset_t | 输入 | Tensor | INT32/INT64 | ND | [B+1] | 序列偏移量（jagged/paged） |
| kv_cache | 输入 | Tensor（paged） | 同 q | ND | [num_pages, 2, page_size, N, D] | KV 缓存 |
| page_offsets / page_ids / last_page_len | 输入 | Tensor（paged） | INT32/INT64 | ND | - | 分页元数据 |
| num_context / num_target | 输入 | Tensor（可选） | INT32/INT64 | ND | [B] | 上下文/目标数量 |
| mask_type | 输入 | int | - | - | - | 0：下三角；2：不使用；3：自定义（1 暂不支持） |
| max_seq_len / max_seq_len_k | 输入 | int | - | - | - | 序列最大长度，[1, 20480] |
| silu_scale | 输入 | float | - | - | - | Silu 缩放，传 0.0 时取 1/max_seq_len |
| target_group_size | 输入 | int | - | - | - | 内置 target mask 分组大小 |
| alpha | 输入 | float（可选） | - | - | - | Alpha 缩放参数 |
| deterministic | 输入 | bool | - | - | - | 确定性计算开关 |
| output | 输出 | Tensor | 同 q（A5 FP8 输入时为 FLOAT16） | ND | [B, S, N_q, D_v] 或 [s_b, N_q, D_v] | 输出张量 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.attention.hstu_*` 时生效。

### 算子约束

- q/k/v 的 B、S、N、D 均不能为 0
- GQA：`N_q % N_k == 0` 且 `N_k == N_v`
- v_dim 需为 16 的倍数，范围 [16, 512]
- mask_type=1（上三角）暂不支持
- max_seq_len 范围 [1, 20480]
- FP8（fp8_e4m3fn）仅在 Atlas 950 上支持

### torch 扩展层额外校验

`ops_rec.ascendc.attention.hstu_*` 接口在调用算子前完成：

- 输入 tensor 必须已定义且非空（q/k/v 必选）
- max_seq_len / mask_type 合法性校验
- mask_type=3 时 mask 必须传入
- 需要安装 `torch_npu`

## 图模式（inductor）支持

四个前向接口均注册了 Meta 实现（`TORCH_LIBRARY_IMPL(ops_rec, Meta, m)`），支持 `torch.compile(..., backend="inductor")` 构图：

```python
compiled = torch.compile(ops_rec.ascendc.attention.hstu_dense, dynamic=True, fullgraph=True)
y = compiled(q, k, v, None, None, 0, 256, 1.0 / 256)
```

## 安装方式

```bash
bash build.sh --install --socs atlas_950 --ops hstu_v1 -j8
pip install dist/ops_rec-*.whl
```

## 调用示例

```python
import torch
import ops_rec

# dense（normal）格式
B, S, N, D = 2, 256, 8, 64
q = torch.randn(B, S, N, D, dtype=torch.float16).npu()
k = torch.randn(B, S, N, D, dtype=torch.float16).npu()
v = torch.randn(B, S, N, D, dtype=torch.float16).npu()
y = ops_rec.ascendc.attention.hstu_dense(q, k, v, None, None, 0, S, 1.0 / S)
print(y.shape)  # torch.Size([2, 256, 8, 64])

# jagged 格式（GQA + dim 不等）
s_b, N_q, D_qk, D_v = 512, 8, 64, 128
q = torch.randn(s_b, N_q, D_qk, dtype=torch.float16).npu()
k = torch.randn(s_b, 2, D_qk, dtype=torch.float16).npu()   # N_k=2（GQA）
v = torch.randn(s_b, 2, D_v, dtype=torch.float16).npu()    # D_v=128（dim 不等）
seq_offsets = torch.tensor([0, 256, 512], dtype=torch.int64).npu()
y = ops_rec.ascendc.attention.hstu_jagged.equal(
    q, k, v, None, None, 0, 256, 1.0 / 256, seq_offsets)
print(y.shape)  # torch.Size([512, 8, 128])
```

## 执行样例

可独立运行的 PyTorch 样例见 [examples/torch/](../examples/torch/)，覆盖以下场景：

| 场景 | 样例文件 |
|------|----------|
| attention 基础（dense） | [test_torch_hstu_forward_dense.py](../examples/torch/test_torch_hstu_forward_dense.py) |
| MHA / GQA（jagged，N_q==N_k 与 N_q>N_k） | [test_torch_hstu_forward_jagged.py](../examples/torch/test_torch_hstu_forward_jagged.py)、[test_torch_hstu_forward_gqa.py](../examples/torch/test_torch_hstu_forward_gqa.py) |
| causal / context / target mask | [test_torch_hstu_forward_mask.py](../examples/torch/test_torch_hstu_forward_mask.py) |
| kvcache（paged 分页解码） | [test_torch_hstu_forward_paged.py](../examples/torch/test_torch_hstu_forward_paged.py) |
| 量化（fp8，Atlas 950） | [test_torch_hstu_forward_fp8.py](../examples/torch/test_torch_hstu_forward_fp8.py) |

运行方式：`python3 examples/torch/test_torch_hstu_forward_<场景>.py`（或 `bash run_examples.sh --type torch --ops hstu_forward_v1` 自动发现执行）。
