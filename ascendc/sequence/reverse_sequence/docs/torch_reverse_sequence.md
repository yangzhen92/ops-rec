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

该接口支持反向传播：reverse 是对合运算，反向等价于对 `grad_output` 再执行一次 reverse（INT64 输入仅支持前向）。

## 接口原型

```python
ops_rec.ascendc.sequence.reverse_sequence(input: Tensor, seq_lengths: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| input | 输入 | Tensor | FLOAT32 / FLOAT16 / BF16 / INT64 | ND | [bs, max_seq_len, data_dim] | 输入序列 |
| seq_lengths | 输入 | Tensor | INT64 / INT32 | ND | [bs] | 每个 batch 的实际序列长度 |
| output | 输出 | Tensor | 与 input 一致 | ND | [bs, max_seq_len, data_dim] | 逆序后的序列 |

## 约束说明

约束分为两层：算子自身语义适用于 aclnn 与 torch 两条调用路径；torch 扩展层额外加的入参校验仅在调用 `ops_rec.ascendc.sequence.reverse_sequence` 时生效。

### 算子约束

- input 必须为 3D，seq_lengths 必须为 1D，且 `input.shape[0] == seq_lengths.shape[0]`
- input 数据类型仅支持 FLOAT32 / FLOAT16 / BF16 / INT64，seq_lengths 仅支持 INT64 / INT32
- 该算子需要在 NPU 设备上运行

### torch 扩展层额外校验

由 [reverse_sequence_plugin.cpp](../torch_extension/reverse_sequence_plugin.cpp) 在调用算子前完成：

- 输入 tensor 必须已定义且非空，且位于同一 NPU 设备
- batch_size ≤ 10240，max_seq_len ≤ 102400，data_dim ∈ [1, 1024]
- 需要安装 `torch_npu`

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_reverse_sequence.py](../examples/torch/test_torch_reverse_sequence.py)。
