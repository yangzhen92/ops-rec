# _group_norm_mul_dropout_fwd

## 来源

`_group_norm_mul_dropout_fwd` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

`_group_norm_mul_dropout_fwd` 是 Group Normalization、逐元素乘法和 Dropout 的前向融合算子。输入
`X` 和 `U` 的形状均为 `[B, Heads, D]`。算子对 `X` 的每个 batch、每个 head 独立地沿
最后一维 `D` 计算均值与方差，然后执行仿射变换并与 `U` 逐元素相乘：

```text
Mean[b, h] = mean(X[b, h, :])
Rstd[b, h] = 1 / sqrt(var(X[b, h, :]) + eps)
Y[b, h, d] = ((X[b, h, d] - Mean[b, h]) * Rstd[b, h] * W[h] + Bias[h]) * U[b, h, d]
```

当 `TRAINING=True` 时，算子会按照 `dropout_ratio` 对输出应用 Dropout。当
`CONCAT_UX=True` 时，输出依次拼接 Dropout 后的 `U`、`X` 和 `Y`。当前 ATK 精度与性能对比固定使用
`TRAINING=False`、`CONCAT_UX=False` 和 `dropout_ratio=0`，避免 GPU 与 NPU 随机数实现差异影响精度结果。

### 输入

- `X` `(B, Heads, D)` — 待归一化的输入特征
- `U` `(B, Heads, D)` — 与归一化结果逐元素相乘的输入
- `W` `(Heads,)` — 每个 head 的仿射缩放参数
- `Bias` `(Heads,)` — 每个 head 的仿射偏置参数

### 输出

- `Y` `(B, Heads, D)` — 当前 ATK 配置下的融合计算结果
- `Mean` `(B, Heads)` — 每个 batch、每个 head 的均值
- `Rstd` `(B, Heads)` — 每个 batch、每个 head 的标准差倒数

当 `CONCAT_UX=True` 时，`Y` 的逻辑形状为 `(B, 3, Heads, D)`，三个分片依次为 `U`、`X` 和融合结果。

### 约束

- `X` 与 `U` 必须是形状完全一致的三维 contiguous tensor
- `W` 与 `Bias` 的形状必须均为 `(Heads,)`
- `X`、`U`、`W` 和 `Bias` 必须使用相同 dtype
- 支持 `fp32`、`fp16`、`bf16`
- `B > 0`、`Heads > 0`、`D > 0`
- 当前用例生成器会将 `D` 调整为不小于原值的最近 2 的幂

## 目录结构

```text
experimental/triton/atk_test/_group_norm_mul_dropout_fwd/
├── README.md
├── gpu_atk_test/
│   ├── _group_norm_mul_dropout_fwd_kernel.py   # GPU 端 Triton kernel 实现
│   └── _group_norm_mul_dropout_fwd_api.py      # GPU 端 ATK API 执行文件
└── npu_atk_test/
    ├── _group_norm_mul_dropout_fwd_kernel.py   # NPU 端 Triton kernel 实现
    ├── _group_norm_mul_dropout_fwd.yaml        # ATK 用例设计（dtype/shape/约束）
    ├── generate_group_norm_mul_dropout_fwd.py  # ATK 用例生成器
    ├── _group_norm_mul_dropout_fwd_api.py      # NPU 端 ATK API 执行文件
    └── _group_norm_mul_dropout_fwd_performance.json    # ATK 生成的用例配置
```

## 调用示例

通过 Python 直接导入调用：

```python
import torch
import triton

from _group_norm_mul_dropout_fwd_kernel import _group_norm_mul_dropout_fwd

B, Heads, D = 8, 32, 128
device = "npu"
x = torch.randn((B, Heads, D), dtype=torch.float32, device=device)
u = torch.randn_like(x)
w = torch.randn((Heads,), dtype=x.dtype, device=device)
bias = torch.randn((Heads,), dtype=x.dtype, device=device)
y = torch.empty_like(x)
mean = torch.empty((B, Heads), dtype=x.dtype, device=device)
rstd = torch.empty((B, Heads), dtype=x.dtype, device=device)

block_h = 16
grid = (B, triton.cdiv(Heads, block_h))
_group_norm_mul_dropout_fwd[grid](
    x,
    u,
    y,
    w,
    bias,
    mean,
    rstd,
    D,
    Heads,
    1e-6,
    42,
    0.0,
    x.stride(0),
    u.stride(0),
    y.stride(0),
    BLOCK_D=triton.next_power_of_2(D),
    BLOCK_H=block_h,
    TRAINING=False,
    CONCAT_UX=False,
)
print(y.shape, mean.shape, rstd.shape)
```

## ATK 测试

测试在 NPU / GPU 两端协同进行：

- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。

### 3.1 远端 NPU 服务器启动监听

```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```

### 3.2 生成测试数据

```bash
atk case -f _group_norm_mul_dropout_fwd.yaml -p generate_group_norm_mul_dropout_fwd.py
```

生成器会统一 `X`、`U`、`W` 和 `Bias` 的 dtype，使 `U` 的 shape 与 `X` 一致，并使
`W`、`Bias` 的长度等于 `X.shape[1]`。

### 3.3 精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all__group_norm_mul_dropout_fwd.json --task accuracy -p _group_norm_mul_dropout_fwd_api.py
```

如需保存精度对比数据，在命令末尾增加 `--save_data profile`。

### 3.4 性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all__group_norm_mul_dropout_fwd.json --task performance_device -p _group_norm_mul_dropout_fwd_api.py
```

如需保存性能对比数据，在命令末尾增加 `--save_data profile`。
