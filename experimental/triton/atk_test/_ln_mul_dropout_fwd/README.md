# `_ln_mul_dropout_fwd` 算子文档

## 来源

`_ln_mul_dropout_fwd` 算子源自 NVIDIA recsys-examples 代码仓：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

`_ln_mul_dropout_fwd` 是基于 Triton 实现的融合前向算子，在一次 kernel 中完成逐行 LayerNorm、仿射变换、与 `u` 逐元素相乘以及可选 dropout，减少中间结果的读写。

对于 `x` 的第 `n` 行，计算过程为：

```text
mean[n] = sum(x[n, d]) / D
rstd[n] = 1 / sqrt(sum((x[n, d] - mean[n])^2) / D + eps)
ln[n, d] = (x[n, d] - mean[n]) * rstd[n] * weight[d] + bias[d]
z[n, d] = ln[n, d] * u_flat[n, d]
```

其中 `u_flat` 为 `u` 按行展平后的 `(N, D)` 视图。`u` 可以是二维张量 `(N, D)`，也可以是三维张量 `(N, H, UD)`，三维情况下要求 `H * UD = D`。

- `training=False` 时不执行 dropout。
- `training=True` 时使用 inverted dropout：保留的元素除以 `1 - dropout_ratio`。
- `concat_ux=False` 时，仅对 `z` 执行可选 dropout，输出形状为 `(N, D)`。
- `concat_ux=True` 时，分别对 `u_flat`、`x` 和 `z` 使用独立随机掩码，并按最后一维拼接，输出形状为 `(N, 3D)`。

## 输入

| 参数 | 说明 |
|------|------|
| `x` | LayerNorm 输入，shape `(N, D)` |
| `u` | 逐元素乘法输入，shape `(N, D)` 或 `(N, H, UD)`；三维时 `H * UD = D` |
| `weight` | LayerNorm 仿射缩放参数，shape `(D,)` |
| `bias` | LayerNorm 仿射偏置参数，shape `(D,)` |
| `eps` | 方差计算的稳定项，要求大于 0 |
| `dropout_ratio` | dropout 概率，取值范围 `[0, 1)`；当前 ATK YAML 用例固定为 `0.0` |
| `training` | 是否启用 dropout |
| `concat_ux` | 是否返回 `u_flat`、`x`、`z` 三部分的拼接结果 |
| `seed` | dropout 随机种子；原生 wrapper 可选，当前 ATK API 会忽略用例中的值并按 `BLOCK_D` 派生固定种子 |

原生 wrapper 要求 `x`、`u`、`weight` 和 `bias` dtype 一致，支持 `fp32`、`fp16` 和 `bf16`。当前 GPU/NPU ATK API 为规避 NPU 编译链中的低精度写回转换问题，会将这四个输入统一转换为 `fp32` 后再调用 kernel，因此 ATK 实际参与对比的输出也是 `fp32`。

## 输出

原生 wrapper `triton_layer_norm_mul_dropout_fwd` 返回：

```text
(y, mean, rstd, BLOCK_D, num_warps, seed)
```

| 返回值 | 说明 |
|--------|------|
| `y` | 主输出；`concat_ux=False` 时 shape `(N, D)`，否则 shape `(N, 3D)`；dtype 与 wrapper 输入 `x` 相同 |
| `mean` | 每行均值，shape `(N,)`，dtype 为 `fp32` |
| `rstd` | 每行标准差倒数，shape `(N,)`，dtype 为 `fp32` |
| `BLOCK_D` | kernel 使用的列块大小，即不超过 64 KiB 特征容量的 `next_power_of_2(D)` |
| `num_warps` | kernel 启动使用的 warp 数 |
| `seed` | 本次执行实际使用的随机种子 |

ATK 自定义 API 仅返回主输出 `y`，`mean`、`rstd` 和启动参数不参与精度对比。

## 调用示例

通过 Python 直接调用原生 wrapper（以下命令在 `gpu_atk_test/` 或 `npu_atk_test/` 目录中执行）：

```python
import torch

from triton_norm_mul_dropout import triton_layer_norm_mul_dropout_fwd

N, D = 128, 512
device = "npu"

x = torch.randn((N, D), dtype=torch.float32, device=device)
u = torch.randn((N, 8, 64), dtype=torch.float32, device=device)
weight = torch.randn((D,), dtype=torch.float32, device=device)
bias = torch.randn((D,), dtype=torch.float32, device=device)

y, mean, rstd, block_d, num_warps, used_seed = (
    triton_layer_norm_mul_dropout_fwd(
        x=x,
        u=u,
        weight=weight,
        bias=bias,
        eps=1e-5,
        dropout_ratio=0.1,
        training=True,
        concat_ux=True,
        seed=2025,
    )
)

print(y.shape)       # torch.Size([128, 1536])
print(mean.shape)    # torch.Size([128])
print(rstd.shape)    # torch.Size([128])
print(block_d)       # 512
print(used_seed)     # 2025
```

如在 CUDA GPU 上运行，将 `device` 改为 `"cuda"`。

## 关键约束

- `x` 必须是二维张量 `(N, D)`。原生 wrapper 对 `N == 0` 提供空输出分支，但当前 ATK 用例要求 `N > 0`。
- `weight` 和 `bias` 必须是一维张量，且元素数均为 `D`。
- 二维 `u` 的 shape 必须为 `(N, D)`；三维 `u` 的 shape 必须为 `(N, H, UD)`，并满足 `H * UD = D`。
- 当前用例生成器仅在 `D`、`H` 和 `UD` 均为 2 的幂时保留三维 `u` 分支；不满足条件的用例会被规范为二维 `(N, D)`。
- 单行特征占用不能超过 64 KiB，即 `D <= 65536 // x.element_size()`；对应上限为 fp32 的 `16384`，以及 fp16/bf16 的 `32768`。
- 原生 wrapper 中各张量 dtype 必须一致。当前 ATK API 无论 YAML 中生成何种浮点 dtype，都会统一转为 fp32 执行。
- `eps > 0`，且 `0 <= dropout_ratio < 1`。当前 ATK YAML 仅测试 `dropout_ratio=0.0`。
- GPU/NPU 精度对比需要相同的随机种子。当前 ATK API 使用 `BLOCK_D` 确定性派生种子，保证两端随机流一致，并忽略 JSON/YAML 中传入的 `seed` 值。

## 目录结构

```text
experimental/triton/atk_test/_ln_mul_dropout_fwd/
├── README.md
├── gpu_atk_test/
│   ├── common.py                         # contiguous 辅助函数
│   ├── triton_norm_mul_dropout.py        # GPU(CUDA) 端 Triton kernel 与 wrapper
│   └── _ln_mul_dropout_fwd_api.py        # GPU 端 ATK 自定义 API
└── npu_atk_test/
    ├── common.py                         # contiguous 辅助函数
    ├── triton_norm_mul_dropout.py        # NPU 端 Triton kernel 与 wrapper
    ├── _ln_mul_dropout_fwd.yaml          # ATK 用例设计
    ├── generate_ln_mul_dropout_fwd.py    # shape/dtype 约束生成器
    ├── _ln_mul_dropout_fwd_api.py        # NPU 端 ATK 自定义 API
    └── all__ln_mul_dropout_fwd.json      # 已生成的 ATK 测试用例
```

## ATK 测试

测试在 GPU 与 NPU 两端协同进行：`gpu_atk_test/` 部署到 GPU 端，`npu_atk_test/` 部署到 NPU 端。以下生成及测试命令均在 NPU 端的 `npu_atk_test/` 目录中执行：

```bash
cd experimental/triton/atk_test/_ln_mul_dropout_fwd/npu_atk_test
```

### 远端 GPU 服务器启动监听

在 GPU 端进入 `gpu_atk_test/` 目录后执行：

```bash
atk server --devices 0 --plugin_path _ln_mul_dropout_fwd_api.py
```

如 GPU server 不在默认地址或端口，请在 NPU 端执行测试时按 ATK 环境配置远端 host/port。

### 生成测试数据

```bash
atk case -f _ln_mul_dropout_fwd.yaml -p generate_ln_mul_dropout_fwd.py
```

生成的精度测试用例通常位于 `result/_ln_mul_dropout_fwd/json/all__ln_mul_dropout_fwd.json`。本目录同时提供可直接复用的 `all__ln_mul_dropout_fwd.json`。

### 精度测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all__ln_mul_dropout_fwd.json --task accuracy \
    -p _ln_mul_dropout_fwd_api.py
```

如需保存精度对比数据，在命令末尾增加 `--save_data profile`。

### 性能测试

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all__ln_mul_dropout_fwd.json --task performance_device \
    -p _ln_mul_dropout_fwd_api.py
```

如需保存性能对比数据，在命令末尾增加 `--save_data profile`。
