# jagged_dense_elementwise_mul_jagged_out_kernel

## 来源

`jagged_dense_elementwise_mul_jagged_out_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_dense_elementwise_mul_jagged_out.py>

## 功能说明

`jagged_dense_elementwise_mul_jagged_out_kernel` 实现 jagged tensor 与 dense 方阵的逐元素相乘（jagged out）：jagged_A 的每个 batch 段是 `seq_len[i] × seq_len[i]` 方阵的展平表示，dense_B 是 `[max_seq_len, max_seq_len]` 共享方阵，每个 batch 取 dense_B 左上 `seq_len[i] × seq_len[i]` 子块与对应 jagged 段逐元素相乘。

### 输入

- `a_ptr` `(Σseq²,)` — jagged 输入，每 batch 为 `seq×seq` 方阵展平
- `b_ptr` `(max_seq_len, max_seq_len)` — dense 方阵，所有 batch 共享
- `a_seq_lengths_ptr` `(B,)` — 每 batch 方阵边长，int32
- `a_offsets_ptr` `(B+1,)` — `seq²` 的 cumsum，int32
- `max_seq_len` — 方阵边长上限，kernel 按 `min(seq_len, max_seq_len)` 截断

### 输出

- `c_ptr` `(Σseq²,)` — 逐元素相乘结果，与 `a_ptr` 同 shape

### 约束

- `a_ptr` 与 `b_ptr` dtype 一致，支持 `fp32`、`fp16`、`bf16`；`b_ptr` 必须是方阵
- `a_seq_lengths_ptr` 长度 B 与 `a_offsets_ptr` 长度 B+1 对应；`a_offsets_ptr` 单调递增，末值 = `a_ptr` 长度
- GPU 端固定 `BLOCK_M = BLOCK_N = 32`；NPU 端通过 `@triton.autotune` 按 `max_seq_len` 自动选择 BLOCK（16~128 档）；NPU 端输入统一转 fp32 编译，输出转回原始 dtype 以对齐双标杆检查

## 目录结构

```text
jagged_dense_elementwise_mul_jagged_out_kernel/
├── README.md
├── gpu_atk_test/
│   ├── jagged_dense_elementwise_mul_jagged_out_kernel.py       # GPU 端 Triton kernel 实现
│   └── jagged_dense_elementwise_mul_jagged_out_kernel_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── jagged_dense_elementwise_mul_jagged_out_kernel.py       # NPU 端 Triton kernel 实现（autotune）
    ├── jagged_dense_elementwise_mul_jagged_out_kernel.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_jagged_dense_elementwise_mul_jagged_out_kernel.py  # ATK 用例生成器
    ├── jagged_dense_elementwise_mul_jagged_out_kernel_api.py   # NPU 端 ATK API 执行文件
    └── jagged_dense_elementwise_mul_jagged_out_kernel_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch
import triton

from jagged_dense_elementwise_mul_jagged_out_kernel import jagged_dense_elementwise_mul_jagged_out_kernel

B, seq_len = 2, 32
total = B * seq_len * seq_len
device = "npu"
a = torch.randn(total, dtype=torch.float32, device=device)
b = torch.randn(seq_len, seq_len, dtype=a.dtype, device=device)
c = torch.zeros(total, dtype=a.dtype, device=device)
lengths = torch.full((B,), seq_len, dtype=torch.int32, device=device)
offsets = torch.tensor([0, seq_len * seq_len, total], dtype=torch.int32, device=device)

BLOCK_M, BLOCK_N = 32, 32
grid = (B, triton.cdiv(seq_len, BLOCK_M))
jagged_dense_elementwise_mul_jagged_out_kernel[grid](
    a, b, c, lengths, offsets,
    a.stride(0), b.stride(0), b.stride(1), seq_len,
    BLOCK_M, BLOCK_N,
)
print(c.shape)  # torch.Size([2048])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/jagged_dense_elementwise_mul_jagged_out_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f jagged_dense_elementwise_mul_jagged_out_kernel.yaml -p generate_jagged_dense_elementwise_mul_jagged_out_kernel.py
```

生成器生成各 batch 方阵边长，设置 `a_ptr = [Σseq²]`、`b_ptr = [max(seq), max(seq)]` 方阵与 `offsets = seq² cumsum`。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_jagged_dense_elementwise_mul_jagged_out_kernel.json --task accuracy \
    -p jagged_dense_elementwise_mul_jagged_out_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c jagged_dense_elementwise_mul_jagged_out_kernel_performance.json --task performance_device \
    -p jagged_dense_elementwise_mul_jagged_out_kernel_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
