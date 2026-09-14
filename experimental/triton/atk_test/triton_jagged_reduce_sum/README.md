# triton_jagged_reduce_sum

## 来源

`jagged_reduce_sum` 算子源自 Meta generative-recommenders：

<https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/ops/triton/triton_jagged.py>

## 功能说明

该算子将一个 Jagged Tensor 按 batch 分段，并对每个分段沿序列维求和。`Jagged` 使用
二维连续存储 `[L, D]`，其中 `L` 是所有 batch 序列长度之和；`seq_offsets` 用前缀和描述
每个 batch 在 `Jagged` 中的起止位置。

对 batch `b` 和特征位置 `d`，计算公式为：

```text
start = seq_offsets[b]
end = seq_offsets[b + 1]
Out[b, d] = sum(Jagged[n, d], n=start..end-1)
```

当 `start == end` 时，该 batch 对应空序列，输出为 `0`。kernel 使用 `fp32` 累加器降低
长序列求和误差，最终再转换为输出 tensor 的 dtype。

### 输入

- `Jagged` `(L, D)`：所有 batch 的变长序列按行拼接后的二维 tensor。
- `seq_offsets` `(B + 1,)`：每个序列的起止偏移，dtype 为 `int64`。

### 输出

- `Out` `(B, D)`：每个 batch 分段沿序列维求和后的结果。

### 约束

- `Jagged` 必须是二维 contiguous tensor，支持 `fp32`、`fp16` 和 `bf16`。
- `seq_offsets` 必须是一维 `int64` tensor，长度至少为 `2`。
- 合法 offsets 满足首元素为 `0`、末元素为 `L`，并且单调不减。
- `D > 0`，允许某些分段长度为 `0`。
- API 根据输入 shape 确定性重建 offsets，保证 NPU 与 GPU 使用相同分段。
- generator 将 `D` 映射到预设候选值，并限制 `(L + B) * D <= 2^24`。

## Kernel 实现

kernel 使用二维 grid：第一维对应 batch，第二维对应特征列块。每个 program 先读取
`seq_offsets[b]` 和 `seq_offsets[b + 1]`，随后循环遍历该分段的所有行，将当前
`BLOCK_D` 个特征累加到 `fp32` accumulator。

GPU 版本按 `D` 计算固定 `BLOCK_D`：当 `D < 64` 时取不小于 `D` 的最近二次幂，否则取
`64`。NPU 版本使用 Triton autotune，在 `16`、`32`、`64` 三种 `BLOCK_D` 中选择配置，
并通过依赖 `BLOCK_D` 的 grid 覆盖完整特征维。

## 目录结构

```text
experimental/triton/atk_test/triton_jagged_reduce_sum/
├── README.md
├── gpu_atk_test/
│   ├── triton_jagged_reduce_sum_kernel.py
│   └── triton_jagged_reduce_sum_api.py
└── npu_atk_test/
    ├── triton_jagged_reduce_sum_kernel.py
    ├── triton_jagged_reduce_sum_api.py
    ├── triton_jagged_reduce_sum.yaml
    └── generate_triton_jagged_reduce_sum.py
```

- `triton_jagged_reduce_sum_kernel.py`：实现分段求和；NPU 文件包含 autotune 配置。
- `triton_jagged_reduce_sum_api.py`：校验输入，重建合法 offsets，创建输出并启动 kernel。
- `triton_jagged_reduce_sum.yaml`：定义随机精度用例的 dtype、shape 和数值范围。
- `generate_triton_jagged_reduce_sum.py`：联动修正 `L`、`D` 和 batch 数，限制内存规模。

当前目录没有单独的 performance JSON，精度和性能任务均使用 case 命令生成的
`all_triton_jagged_reduce_sum.json`。

## 调用示例

下面示例直接调用 GPU 端 kernel；NPU 端通常通过 API 使用 autotune 版本。

```python
import torch
import triton

from triton_jagged_reduce_sum_kernel import jagged_reduce_sum

device = "cuda"
jagged = torch.randn((9, 64), device=device, dtype=torch.float32)
seq_offsets = torch.tensor([0, 2, 2, 6, 9], device=device, dtype=torch.int64)

batch_size = seq_offsets.numel() - 1
d = jagged.shape[1]
out = torch.empty((batch_size, d), device=device, dtype=jagged.dtype)
block_d = triton.next_power_of_2(d) if d < 64 else 64
grid = (batch_size, triton.cdiv(d, block_d))

jagged_reduce_sum[grid](
    seq_offsets=seq_offsets,
    Jagged=jagged,
    Out=out,
    D=d,
    stride_jn=jagged.stride(0),
    stride_ob=out.stride(0),
    BLOCK_D=block_d,
)
print(out.shape)
```

## ATK 测试

测试由 GPU server 与 NPU client 协同完成。GPU 端部署 `gpu_atk_test/`，NPU 端部署
`npu_atk_test/`。

### 1. NPU 端生成精度用例

```bash
atk case -f triton_jagged_reduce_sum.yaml -p generate_triton_jagged_reduce_sum.py
```

该命令按照 YAML 采样输入，再由 generator 修正 `Jagged` 与 `seq_offsets` 的 shape，生成
`all_triton_jagged_reduce_sum.json`。

### 2. GPU 端启动 ATK 服务

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

该命令在 GPU 设备 `0` 上启动标杆服务，并从 `gpu_atk_test` 加载 API 和 kernel。

### 3. NPU 端执行精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_jagged_reduce_sum.json --task accuracy -p triton_jagged_reduce_sum_api.py
```

该命令在本地 NPU 和指定的远端 GPU 上执行相同用例，并比较 `Out` 的数值结果。需要保存
中间数据时，可在命令末尾增加 `--save_data profile`。

### 4. NPU 端执行性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_jagged_reduce_sum.json --task performance_device -p triton_jagged_reduce_sum_api.py
```

该命令复用生成的完整用例集，分别测量 NPU 与 GPU 的设备执行时间。
