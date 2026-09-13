# triton_ln_mul_dropout_bwd_dx_du

## 来源

`_ln_mul_dropout_bwd_dx_du` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

该算子计算 LayerNorm、逐元素乘法和 Dropout 融合前向过程对 `X` 与 `U` 的反向梯度。
令 `x_hat = (X - Mean) * Rstd`，关闭 Dropout 与拼接路径时，主要计算为：

```text
ln = x_hat * W + B
dU = dY * ln
dLN = dY * U
dX = Rstd * LayerNormBackward(dLN * W, x_hat)
```

kernel 还会生成 partial `DW/DB`，供 `_ln_mul_dropout_bwd_dwdb` 后续规约。当前 ATK API
设置 `COMPUTE_Y=False`，只返回并比较 `DX` 与 `DU`。

### 输入

- `X` `(N, D)`：LayerNorm 前向输入。
- `U` `(N, D)`：与归一化结果逐元素相乘的输入。
- `DY` `(N, D)`：上游梯度。
- `W` `(D,)`：可选缩放参数；未提供时使用单位权重。
- `B` `(D,)`：可选偏置参数；未提供时使用零偏置。

`Mean`、`Rstd`、Dropout seed 和临时 partial gradient 缓冲区由 API 准备。

### 输出

- `DX` `(N, D)`：`X` 的梯度。
- `DU` `(N, D)`：`U` 的梯度。

### 约束

- 当前 generator 覆盖二维 `U`，并使 `X`、`U`、`DY` shape 完全一致。
- `W` 与 `B` 存在时必须是一维 tensor，长度等于 `D`。
- 输入统一为 `fp32`、`fp16` 或 `bf16`。
- 特征维 `D` 最大为 `4096`，总元素数受 generator 限制以避免 OOM。

## 目录结构

```text
experimental/triton/atk_test/triton_ln_mul_dropout_bwd_dx_du/
├── README.md
├── gpu_atk_test/
│   ├── triton_ln_mul_dropout_bwd_dx_du_kernel.py
│   └── triton_ln_mul_dropout_bwd_dx_du_api.py
└── npu_atk_test/
    ├── triton_ln_mul_dropout_bwd_dx_du_kernel.py
    ├── triton_ln_mul_dropout_bwd_dx_du_api.py
    ├── triton_ln_mul_dropout_bwd_dx_du.yaml
    ├── generate_triton_ln_mul_dropout_bwd_dx_du.py
    └── triton_ln_mul_dropout_bwd_dx_du_performance.json
```

- `*_kernel.py`：GPU/NPU Triton 反向 kernel；NPU 版本使用 autotune 配置。
- `*_api.py`：计算 `Mean/Rstd`，准备输出与 partial gradient 缓冲区并调用 kernel。
- `*.yaml`：定义随机测试输入及边界范围。
- `generate_*.py`：对齐 `X/U/DY/W/B` 的 shape 和 dtype。
- `*_performance.json`：保存固定性能用例。

## 调用示例

```python
from triton_ln_mul_dropout_bwd_dx_du_api import LnMulDropoutBwdDxDuApi

api = LnMulDropoutBwdDxDuApi()
api.init_by_input_data(input_data)
dx, du = api()
```

`input_data` 由 ATK 构造。直接使用 API 可复用与测试一致的均值、方差和启动参数准备逻辑。

## ATK 测试

### 1. GPU 端启动服务

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

该命令在 GPU 设备 `0` 启动标杆服务，并加载 GPU API 与 Triton kernel。

### 2. NPU 端生成精度用例

```bash
atk case -f triton_ln_mul_dropout_bwd_dx_du.yaml -p generate_triton_ln_mul_dropout_bwd_dx_du.py
```

该命令根据 YAML 采样用例，再由 generator 对齐相关 tensor，生成
`all_triton_ln_mul_dropout_bwd_dx_du.json`。

### 3. NPU 端执行精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_ln_mul_dropout_bwd_dx_du.json --task accuracy -p triton_ln_mul_dropout_bwd_dx_du_api.py
```

该命令执行 NPU 与远端 GPU 实现并比较 `DX/DU`，用于发现数值精度差异。

### 4. NPU 端执行性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_ln_mul_dropout_bwd_dx_du_performance.json --task performance_device -p triton_ln_mul_dropout_bwd_dx_du_api.py
```

该命令使用固定性能用例测量 NPU 和 GPU 执行时间。需要保留中间数据时，可在任务命令末尾增加
`--save_data profile`。
