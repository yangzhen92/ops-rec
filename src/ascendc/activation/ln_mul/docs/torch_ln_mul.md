# ln_mul

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

对输入 X 进行 LayerNorm，再将结果与 gamma、beta 计算，最后与输入 U 做乘法。

计算公式：

```text
output = LayerNorm(X) * U
LayerNorm(X) = rstd * (x - mean) * gamma + beta
```

## 接口原型

```python
ops_rec.ascendc.activation.ln_mul(x: Tensor, u: Tensor, gamma: Tensor, beta: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| x | 输入 | Tensor | FLOAT32 | ND | [A, R] | 输入数据 |
| u | 输入 | Tensor | FLOAT32 | ND | [A, R] | 输入数据 |
| gamma | 输入 | Tensor | FLOAT32 | ND | [R] | LayerNorm 权重 |
| beta | 输入 | Tensor | FLOAT32 | ND | [R] | LayerNorm 偏置 |
| output | 输出 | Tensor | FLOAT32 | ND | [A, R] | 输出数据 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.activation.ln_mul` 时生效。

### 算子约束

- 仅支持 FLOAT32 数据类型
- X、U 必须为 2D 且 shape 相同
- gamma、beta 必须为 1D
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [ln_mul_plugin.cpp](../torch_extension/ln_mul_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空
- 所有输入 tensor 必须位于同一 NPU 设备
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install
```

## 调用示例

完整调用示例参考 [test_torch_ln_mul.py](../examples/torch/test_torch_ln_mul.py)。
