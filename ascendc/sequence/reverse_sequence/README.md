# reverse_sequence

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
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

算子采用 AscendC 进行开发，将 `batch_size * max_seq_len` 个序列行均摊到各 AI Vector 核处理。每个核按行执行：CopyIn（含 32B 非对齐尾部 DataCopyPad 补齐）→ 计算该行所属 batch 及行号 → 若行号小于该 batch 的 `seq_len`，则将数据 CopyOut 到逆序目标行；否则不写入（该行由调用侧预置的 input 拷贝保留原值）。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| input | 输入 | Tensor | FLOAT32 / FLOAT16 / BF16 / INT64 | ND | [bs, max_seq_len, data_dim] | 输入序列 |
| seq_lengths | 输入 | Tensor | INT64 / INT32 | ND | [bs] | 每个 batch 的实际序列长度 |
| output | 输出 | Tensor | 与 input 一致 | ND | [bs, max_seq_len, data_dim] | 逆序后的序列 |

## 约束说明

- input 必须为 3D，seq_lengths 必须为 1D，且 `input.shape[0] == seq_lengths.shape[0]`
- input 数据类型仅支持 FLOAT32 / FLOAT16 / BF16 / INT64，seq_lengths 仅支持 INT64 / INT32
- **aclnn 路径下 output 必须先用 input 的数据初始化**（kernel 仅覆写每个 batch 前 seq_len 行，其余行保持 output 原值）；torch 接口内部已通过 `input.clone()` 处理
- 该算子需要在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：batch_size ≤ 10240、max_seq_len ≤ 102400、data_dim ∈ [1, 1024]、输入 tensor 非空且位于同一 NPU 设备。详见 [reverse_sequence 接口文档](./docs/torch_reverse_sequence.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_reverse_sequence.cpp)
- [torch 调用示例](./examples/torch/test_torch_reverse_sequence.py)
- [aclnn 接口文档](./docs/aclnn_reverse_sequence.md)
- [torch 接口文档](./docs/torch_reverse_sequence.md)
