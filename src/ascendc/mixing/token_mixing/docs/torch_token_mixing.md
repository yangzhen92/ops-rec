# token_mixing（torch）

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

实现x与其转置x_t相加后的归一化操作。
对输入 `x` 与其转置 `x_t` 求和后做 LayerNorm（带 `gamma` / `beta` / `epsilon`）。

计算公式：

```text
x_t = permute(x, [0, 2, 1])          # [B, S, H] -> [B, H, S]
y   = LayerNorm(x + x_t)             # 沿最后一维
LayerNorm(z) = rstd * (z - mean) * gamma + beta
```

## 接口原型

```python
ops_rec.ascendc.mixing.token_mixing(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    epsilon: float = 1e-7,
) -> Tensor
```

底层注册：

```text
torch.ops.ops_rec.token_mixing(Tensor x, Tensor gamma, Tensor beta, float epsilon=1e-7) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| x | 输入 | Tensor | FLOAT32 | ND | [B, S, H] | 输入数据，仅支持三维张量S,H维度一致 |
| gamma | 输入 | Tensor | FLOAT32 | ND | [H] | LayerNorm 权重，仅支持一维张量 |
| beta | 输入 | Tensor | FLOAT32 | ND | [H] | LayerNorm 偏置，仅支持一维张量 |
| epsilon | 输入 (属性) | float | — | — | — |极小数，防除零，默认 `1e-7` |
| y | 输出 | Tensor | FLOAT32 | ND | [B, H, S] | 输出数据，融合算子归一化结果张量 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.mixing.token_mixing` 时生效。

### 算子约束

- 仅支持 FLOAT32 数据类型
- `x` 必须为 3D，shape 为 `[B, S, H]`；`gamma` / `beta` 必须为 1D，长度等于 `H`
- 转置混合在 `S == H` 时语义正确（kernel 按 `[B*S, H]` 行布局读写 `x` / `x_t`）
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [token_mixing_plugin.cpp](../torch_extension/token_mixing_plugin.cpp) 中 `token_mixing_impl_npu` 在调用 `aclnnTokenMixing` 前完成：

- 输入 tensor 必须已定义且非空（`check_tensor_non_empty`：`x` / `gamma` / `beta`）
- 维数校验（`check_tensor_dim`：`x` 为 3 维，`gamma` / `beta` 为 1 维）
- 所有输入 tensor 必须位于同一 NPU 设备（`check_tensor_npu_device`）
- 需要安装 `torch_npu`

> Meta 路径（`token_mixing_meta`，供 `torch.compile` / inductor 构图）仅做维数 `TORCH_CHECK`，不执行上述非空/同卡校验，也不真正计算。

## 安装方式

```bash
bash build.sh --install
```

## 调用示例

完整调用示例参考 [test_torch_token_mixing.py](../examples/torch/test_torch_token_mixing.py)。
