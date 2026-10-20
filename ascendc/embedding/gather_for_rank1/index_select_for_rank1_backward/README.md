# index_select_for_rank1_backward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

`gather_for_rank1`（1D 嵌入权重表按 index 查表，`y = x[index]`）的反向算子：把每个位置的上游梯度 `grad_y[i]` 按 `index[i]` 散射累加回权重表，即 scatter-add：

```text
grad_x = zeros_like(x)
grad_x[index[i]] += grad_y[i]    (i = 0 .. len(index)-1)
```

该算子与 `gather_for_rank1`（前向）组成一个 autograd 配对（前向保存 `{x, index}`，反向调用本算子）。前向 vendor `gather_for_rank1_forward` 迁移完成后，将通过 `ops_rec/autograd/` 统一注册前反向绑定。

torch 接口返回 `(grad_x, grad_index)`，其中 `grad_index` 为 index 的梯度占位、恒为 undefined（index 不可导），保持与 RecSDK `torch.ops.mxrec.index_select_for_rank1_backward` 一致的解包语义。

对应接口：

- aclnn 接口：`aclnnIndexSelectForRank1Backward`
- torch 接口：`ops_rec.ascendc.embedding.index_select_for_rank1_backward`

## 算子实现原理

算子采用 AscendC 开发，在 AIV 上执行：

1. **区间均摊**：index 元素总数均摊到全部 AIV 核（`baseLen = totalLen / coreNum`），余数由前 `tailSplitIndex` 个核各多处理 1 个元素；
2. **分批搬入**：每核把自己负责的 index/grad_y 区间按 tiling 计算的 `stride` 分批经 `DataCopy + DataCopyPad`（尾部不齐 32B 的部分）搬入 UB；
3. **UB 内 scatter-add**：每核在 UB 内维护全量 `grad_x` 缓冲（先 `Duplicate` 清零），逐元素执行 `gradXUb[index[i]] += gradYUb[i]`；
4. **AtomicAdd 写回**：计算完成后将整个 `grad_x` 缓冲经 `SetAtomicAdd` 整块写回 GM，多核份量自然累加。因此 **grad_x 必须以全零初始化**（torch 层内部通过 `at::zeros_like(x)` 处理，aclnn 路径需调用侧保证）。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| grad_y | 输入 | Tensor | FLOAT32 | ND | [num_index] | 上游梯度 |
| x | 输入 | Tensor | FLOAT32 | ND | [x_dim0] | 前向权重表（仅使用其 shape，数值不参与计算） |
| index | 输入 | Tensor | INT32 / INT64 | ND | [num_index] | 查表索引，与 grad_y 等长 |
| grad_x | 输出 | Tensor | FLOAT32 | ND | [x_dim0] | 权重表梯度，scatter-add 累加结果 |

## 约束说明

- `grad_y` / `x` / `index` 必须均为 1D，且 `grad_y.shape[0] == index.shape[0]`
- `grad_y` / `x` / `grad_x` 仅支持 FLOAT32，`index` 仅支持 INT32 / INT64
- `index` 元素取值必须在 `[0, x.shape[0])` 内（越界为未定义行为，调用侧保证）
- **grad_x 必须以全零初始化**：kernel 通过 AtomicAdd 向 grad_x 累加。aclnn 路径需调用侧保证清零；torch 接口内部已通过 `at::zeros_like(x)` 处理
- kernel 需将 grad_x 全量缓存进 UB，`x.shape[0] * 4B` 加上 20KB 预留须不超过单核 UB 容量
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 非空、维度为 1D、位于同一 NPU 设备、grad_y 与 index 等长。详见 [index_select_for_rank1_backward 接口文档](./docs/torch_index_select_for_rank1_backward.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_index_select_for_rank1_backward.cpp)
- [torch 调用示例](./examples/torch/test_torch_index_select_for_rank1_backward.py)
- [aclnn 接口文档](./docs/aclnn_index_select_for_rank1_backward.md)
- [torch 接口文档](./docs/torch_index_select_for_rank1_backward.md)
