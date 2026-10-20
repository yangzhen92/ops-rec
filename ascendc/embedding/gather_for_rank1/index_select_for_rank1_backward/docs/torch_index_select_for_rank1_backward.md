# index_select_for_rank1_backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

`gather_for_rank1`（1D 嵌入权重表按 index 查表，`y = x[index]`）的反向算子：把每个位置的上游梯度 `grad_y[i]` 按 `index[i]` 散射累加回权重表，等价于：

```python
grad_x = torch.zeros_like(x)
grad_x.scatter_add_(0, index, grad_y)
```

torch 接口返回 `(grad_x, grad_index)`，其中 `grad_index` 为 index 的梯度占位、恒为 `None`（index 不可导），与 RecSDK `torch.ops.mxrec.index_select_for_rank1_backward` 解包语义一致。前向算子 `gather_for_rank1` 迁移完成后，将在 `ops_rec/autograd/` 注册前反向绑定，训练时直接调用前向即可自动微分。

## 接口原型

```python
ops_rec.ascendc.embedding.index_select_for_rank1_backward(grad_y: Tensor, x: Tensor, index: Tensor)
    -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| grad_y | 输入 | Tensor | FLOAT32 | ND | [num_index] | 上游梯度 |
| x | 输入 | Tensor | FLOAT32 | ND | [x_dim0] | 前向权重表（仅使用其 shape，数值不参与计算） |
| index | 输入 | Tensor | INT32 / INT64 | ND | [num_index] | 查表索引，与 grad_y 等长 |
| grad_x | 输出 | Tensor | FLOAT32 | ND | [x_dim0] | 权重表梯度，scatter-add 累加结果 |
| grad_index | 输出 | Tensor | - | - | - | index 梯度占位，恒为 undefined（None） |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.embedding.index_select_for_rank1_backward` 时生效。

### 算子约束

- `grad_y` / `x` / `index` 必须均为 1D，且 `grad_y.shape[0] == index.shape[0]`
- `grad_y` / `x` / `grad_x` 仅支持 FLOAT32，`index` 仅支持 INT32 / INT64
- `index` 元素取值必须在 `[0, x.shape[0])` 内（越界为未定义行为，调用侧保证）
- kernel 需将 grad_x 全量缓存进 UB，`x.shape[0] * 4B` 加上 20KB 预留须不超过单核 UB 容量
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [index_select_for_rank1_backward_plugin.cpp](../../torch_extension/index_select_for_rank1_backward_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空，且位于同一 NPU 设备
- `grad_y` / `x` / `index` 必须均为 1D
- `grad_y.shape[0] == index.shape[0]`
- 需要安装 `torch_npu`

> 接口内部以 `at::zeros_like(x)` 初始化 grad_x（kernel 通过 AtomicAdd 累加），调用侧无需预清零。

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_index_select_for_rank1_backward.py](../examples/torch/test_torch_index_select_for_rank1_backward.py)。
