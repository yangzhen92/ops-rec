# jagged_2_softmax_kernel

## 来源

`jagged_2_softmax_kernel` 算子源自 PyTorch FBGEMM 代码仓：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_softmax.py>

## 功能说明

`jagged_2_softmax_kernel` 对 jagged 方阵做 softmax：输入 `input_x` 为 `[Σseq²]` 的展平 1D tensor，每个 batch 对应一个 `seq×seq` 方阵；kernel 按 `transpose` 指定的方向对每个 sample、每个 head 计算 softmax，输出与输入同 shape。

### 输入

- `input_x` `(Σseq²,)` — 每 batch 为 `seq×seq` 方阵展平
- `offsets_row` / `offsets_col` `(B+1,)` — 方阵边长（`seq`）的 cumsum，int32（方阵 row=col，两者相同）
- `offsets_overall` `(B+1,)` — `seq²` 的 cumsum，int32，共 `B+1` 个元素
- `transpose` — 0/1 标量：1 按行（seq 维）做 softmax，0 按列（head 维）做

### 输出

- `output_y` `(Σseq²,)` — softmax 结果，与 `input_x` 同 shape

### 约束

- `B ∈ {1, 3, 5, 7, 10, 15}`；`seq_len ∈ [16, 256]`，由 `B` 确定性生成（Generator 与 API 逐字一致）
- 三个 `offsets` 长度必须一致（`B+1`）；`input_x` 长度 = `Σseq²`
- `BLOCK_SIZE` 由 API 按 `max(seq_len)` 计算并对上限做 128 限制（限制 UB 占用）；`grid = (B, max_seq_len_col)`
- GPU 端与 NPU 端使用同一 kernel 实现；offset 值由 API 重建，不依赖 ATK 传入值

## 目录结构

```text
jagged_2_softmax_kernel/
├── README.md
├── gpu_atk_test/
│   ├── jagged_2_softmax_kernel.py       # GPU 端 Triton kernel 实现
│   └── jagged_2_softmax_kernel_api.py   # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── jagged_2_softmax_kernel.py       # NPU 端 Triton kernel 实现
    ├── jagged_2_softmax_kernel.yaml     # ATK 用例设计（dtype/shape/约束）
    ├── generate_jagged_2_softmax_kernel.py  # ATK 用例生成器
    ├── jagged_2_softmax_kernel_api.py   # NPU 端 ATK API 执行文件
    └── jagged_2_softmax_kernel_performance.json  # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用（与 ATK API 中的用法一致）：

```python
import torch

from jagged_2_softmax_kernel import jagged_2_softmax_kernel

B, seq_len = 2, 32
total = B * seq_len * seq_len
device = "npu"
x = torch.randn(total, dtype=torch.float32, device=device)
y = torch.zeros(total, dtype=x.dtype, device=device)
offsets_row = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32, device=device)
offsets_overall = torch.tensor([0, seq_len * seq_len, total], dtype=torch.int32, device=device)

BLOCK_SIZE = 32
grid = (B, seq_len)
jagged_2_softmax_kernel[grid](
    x,
    y,
    offsets_row,
    offsets_row.clone(),
    offsets_overall,
    x.stride(0),
    y.stride(0),
    1,  # transpose：按行 softmax
    seq_len,  # max_seq_len_row
    seq_len,  # max_seq_len_col
    BLOCK_SIZE,
)
print(y.shape)  # torch.Size([2048])
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- 精度测试用例在 NPU 端生成。
- 性能测试用例复用 `npu_atk_test/jagged_2_softmax_kernel_performance.json`。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 生成测试数据

```bash
atk case -f jagged_2_softmax_kernel.yaml -p generate_jagged_2_softmax_kernel.py
```

生成器按 `B` 确定性生成 `seq_len`（与 API 逐字一致），设置 `input_x = [Σseq²]`、三个 `offsets = [B+1]`；运行时 offset 值由 API 重建，保证两端一致。

### 远端 GPU 服务器起监听命令

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c all_jagged_2_softmax_kernel.json --task accuracy \
    -p jagged_2_softmax_kernel_api.py
```

### 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 \
    task -c jagged_2_softmax_kernel_performance.json --task performance_device \
    -p jagged_2_softmax_kernel_api.py
```

注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
