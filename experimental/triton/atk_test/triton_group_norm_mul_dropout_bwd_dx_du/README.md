# triton_group_norm_mul_dropout_bwd_dx_du

## 来源

`_group_norm_mul_dropout_bwd_dx_du` 算子源自 NVIDIA recsys-examples：

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_norm_mul_dropout.py>

## 功能说明

该算子计算 Group Normalization、逐元素乘法和 Dropout 融合前向过程对 `X` 与 `U` 的反向梯度。
当前 ATK 封装固定 `TRAINING=False`、`CONCAT_UX=False`、`dropout_ratio=0` 和
`COMPUTE_Y=False`，因此测试覆盖无随机 Dropout、无拼接输出的确定性路径。

对第 `n` 行、第 `h` 个 head，令：

```text
x_hat = (X - Mean) * Rstd
ln = x_hat * W + B
dU = dY * ln
dLN = dY * U
dX = Rstd * W * LayerNormBackward(dLN, x_hat)
```

kernel 同时生成供后续权重梯度规约使用的 partial `DW` 和 `DB`，但本目录的 ATK API
只返回并比较 `DX`、`DU`。

### 输入

- `X` `(N, Heads * D)`：前向输入。
- `U` `(N, Heads * D)`：与归一化结果逐元素相乘的输入。
- `W` `(Heads,)`：每个 head 的缩放参数。
- `B` `(Heads,)`：每个 head 的偏置参数。
- `DY` `(N, Heads * D)`：上游梯度。
- `eps`：归一化稳定项。
- `dropout_ratio`、`TRAINING`、`CONCAT_UX`：融合路径控制参数，当前测试固定为关闭状态。

### 输出

- `DX` `(N, Heads * D)`：`X` 的梯度。
- `DU` `(N, Heads * D)`：`U` 的梯度。

### 约束

- `X`、`U` 与 `DY` 必须具有相同的二维 shape。
- `X.shape[1]` 必须能被 `Heads` 整除。
- `W` 与 `B` 的长度必须等于 `Heads`。
- 输入统一为 `fp32`、`fp16` 或 `bf16`。
- 生成器将特征总维限制在 `4096` 以内，并控制总元素数以避免用例占用过多内存。

## 目录结构

```text
experimental/triton/atk_test/triton_group_norm_mul_dropout_bwd_dx_du/
├── README.md
├── gpu_atk_test/
│   ├── triton_group_norm_mul_dropout_bwd_dx_du_kernel.py
│   └── triton_group_norm_mul_dropout_bwd_dx_du_api.py
└── npu_atk_test/
    ├── triton_group_norm_mul_dropout_bwd_dx_du_kernel.py
    ├── triton_group_norm_mul_dropout_bwd_dx_du_api.py
    ├── triton_group_norm_mul_dropout_bwd_dx_du.yaml
    ├── generate_triton_group_norm_mul_dropout_bwd_dx_du.py
    └── triton_group_norm_mul_dropout_bwd_dx_du_performance.json
```

- `*_kernel.py`：GPU/NPU 端 Triton kernel 定义；NPU 版本包含可执行配置筛选和 autotune 配置。
- `*_api.py`：把 ATK 输入转换为 kernel 参数，准备均值、标准差倒数和临时缓冲区，并返回 `DX/DU`。
- `*.yaml`：定义随机用例的 dtype、shape、数值范围和属性范围。
- `generate_*.py`：对 ATK 独立生成的输入进行 shape 与 dtype 联动修正。
- `*_performance.json`：固定性能用例集合。

## 调用示例

通常应通过同目录的 `triton_group_norm_mul_dropout_bwd_dx_du_api.py` 调用，API 会构造
`Mean`、`Rstd`、partial `DW/DB` 和 kernel 启动参数：

```python
from triton_group_norm_mul_dropout_bwd_dx_du_api import TritonGroupNormMulDropoutBwdDxDuApi

api = TritonGroupNormMulDropoutBwdDxDuApi()
api.init_by_input_data(input_data)
dx, du = api()
```

## ATK 测试

测试由 GPU server 与 NPU client 协同完成。GPU 端使用 `gpu_atk_test/`，NPU 端使用
`npu_atk_test/`。

### 1. GPU 端启动服务

```bash
atk server --devices 0 --plugin_path /path/to/gpu_atk_test
```

该命令在 GPU 设备 `0` 上启动 ATK 服务，并从 `gpu_atk_test` 加载 GPU API 与 kernel。

### 2. NPU 端生成精度用例

```bash
atk case -f triton_group_norm_mul_dropout_bwd_dx_du.yaml -p generate_triton_group_norm_mul_dropout_bwd_dx_du.py
```

该命令按 YAML 采样输入，再由 generator 统一 `X/U/DY`、`W/B` 的 shape 和 dtype，生成
`all_triton_group_norm_mul_dropout_bwd_dx_du.json`。

### 3. NPU 端执行精度对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c all_triton_group_norm_mul_dropout_bwd_dx_du.json --task accuracy -p triton_group_norm_mul_dropout_bwd_dx_du_api.py
```

该命令在 NPU 设备 `0` 执行本地实现，并连接指定 GPU 服务执行标杆实现，然后比较 `DX/DU`。

### 4. NPU 端执行性能对比

```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {gpu端端口} --devices 0 task -c triton_group_norm_mul_dropout_bwd_dx_du_performance.json --task performance_device -p triton_group_norm_mul_dropout_bwd_dx_du_api.py
```

该命令使用固定性能用例分别测量 NPU 与 GPU 设备执行时间。需要保留中间数据时，可在精度或性能
命令末尾增加 `--save_data profile`。
