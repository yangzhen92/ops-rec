# reverse_sequence

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

对输入序列 `input` 的每个 batch，按 `seq_lengths` 指定的实际长度，将前 `seq_len` 个序列行逆序排列，其余行保持原值不变。

计算公式（对每个 batch b）：

```text
output[b, i, :] = input[b, seq_len[b] - 1 - i, :]   (i < seq_len[b])
output[b, i, :] = input[b, i, :]                    (i >= seq_len[b])
```

对应接口：

- aclnn 接口：`aclnnReverseSequenceCust`
- torch 接口：`ops_rec.ascendc.sequence.reverse_sequence`

> GE 算子类型名为 `ReverseSequenceCust`：CANN 内置存在同名算子 `ReverseSequence`，自定义算子沿用该名会在运行时与内置实现冲突（tiling 阶段解析到内置版本导致进程崩溃），故加 `Cust` 后缀规避。

## 算子实现原理

算子采用 AscendC 进行开发，将 `batch_size * max_seq_len` 个序列行均摊到 AI Vector 核处理（启动核数按 `min(AIV核数, 总行数)` 裁剪，余数行由前 `left` 个核各多承担 1 行）。每个核对自己覆盖的连续行区间按 batch 分段处理：

1. **seq_lengths 缓存**：本核覆盖的 batch 区间的 seq_lengths 经 `DataCopyPad` 一次搬入 UB（batch 数超 2048 时分 chunk 多次），之后按 batch 直接读 UB，避免逐行标量读 GM；
2. **有效行筛选**：读取本 batch 的 `seq_len`（超出 `max_seq_len` 时截断），仅行号 < `seq_len` 的行需要搬运，其余行整段跳过、不做任何读写，由调用侧预置的 input 拷贝保留原值；
3. **批量 CopyIn**：有效行在 GM 上连续，以 `DataCopyPad` blockCount 模式批量搬入 UB，UB 内每行按 32B 对齐 stride 存放。按 shape 分两种模式：
   - **span 模式（小 shape）**：单核行区间可整个放入 UB（`行数 ≤ rowsPerTile`）时，一次搬入整个连续行区间（无效行只读不写），全程仅 seq_lengths 就绪一次流水同步；
   - **tile 模式（大 shape）**：按 tile（至多 `rowsPerTile` 行）逐 tile 搬入，单 buffer 下相邻 tile 间串行；
4. **逐行 CopyOut**：每行经 `DataCopyPad` 写到逆序目标行（`batch * max_seq_len + seq_len - 1 - 行号`），GM 侧非对齐地址由 `DataCopyPad` 自动处理；span 模式下 UB 无复用，不需要 MTE3→MTE2 回同步。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| input | 输入 | Tensor | FLOAT32 / FLOAT16 / BF16 / INT64 | ND | [bs, max_seq_len, data_dim] | 输入序列 |
| seq_lengths | 输入 | Tensor | INT64 / INT32 | ND | [bs] | 每个 batch 的实际序列长度 |
| output | 输出 | Tensor | 与 input 一致 | ND | [bs, max_seq_len, data_dim] | 逆序后的序列 |

## 约束说明

- input 必须为 3D，seq_lengths 必须为 1D，且 `input.shape[0] == seq_lengths.shape[0]`
- input 数据类型仅支持 FLOAT32 / FLOAT16 / BF16 / INT64，seq_lengths 仅支持 INT64 / INT32
- seq_lengths 元素取值应在 [0, max_seq_len] 范围内；越界值由 kernel 防御性截断（大于 max_seq_len 按 max_seq_len 处理），负值等效于 0（不逆序）
- **aclnn 路径下 output 必须先用 input 的数据初始化**（kernel 仅覆写每个 batch 前 seq_len 行，其余行保持 output 原值）；torch 接口内部已通过 `input.clone()` 处理
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：batch_size ≤ 10240、max_seq_len ≤ 102400、data_dim ∈ [1, 1024]、输入 tensor 非空且位于同一 NPU 设备。详见 [reverse_sequence 接口文档](./docs/torch_reverse_sequence.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_reverse_sequence.cpp)
- [torch 调用示例](./examples/torch/test_torch_reverse_sequence.py)
- [aclnn 接口文档](./docs/aclnn_reverse_sequence.md)
- [torch 接口文档](./docs/torch_reverse_sequence.md)
