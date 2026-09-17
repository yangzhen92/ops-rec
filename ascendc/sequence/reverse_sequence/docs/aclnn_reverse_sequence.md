# aclnnReverseSequenceCust

> GE 算子类型名为 `ReverseSequenceCust`：CANN 内置存在同名算子 `ReverseSequence`，自定义算子沿用该名会在运行时与内置实现冲突（tiling 阶段解析到内置版本导致进程崩溃），故加 `Cust` 后缀规避。

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √ |
| Atlas A2系列产品 | √ |

## 功能说明

对输入序列 `input` 的每个 batch，按 `seq_lengths` 指定的实际长度，将前 `seq_len` 个序列行逆序写入 `output`，其余行不写入（保持 `output` 原值）。

计算公式（对每个 batch b）：

```text
output[b, i, :] = input[b, seq_len[b] - 1 - i, :]   (i < seq_len[b])
output[b, i, :] = output[b, i, :]                   (i >= seq_len[b]，kernel 不写入)
```

> 为得到"其余行保持 input 原值"的完整语义，调用前需用 input 的数据初始化 output（等价于 torch 层的 `output = input.clone()`）。

## 函数原型

两段式接口，必须先调用 `aclnnReverseSequenceCustGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnReverseSequenceCust` 接口执行计算。

```c
aclnnStatus aclnnReverseSequenceCustGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *seqLengths,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnReverseSequenceCust(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnReverseSequenceCustGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| input（aclTensor） | 输入 | 输入序列，shape [bs, max_seq_len, data_dim] | FLOAT32 / FLOAT16 / BF16 / INT64 | ND |
| seqLengths（aclTensor） | 输入 | 每个 batch 的实际序列长度，shape [bs] | INT64 / INT32 | ND |
| out（aclTensor） | 输出 | 逆序后的序列，shape [bs, max_seq_len, data_dim]，数据类型与 input 一致 | FLOAT32 / FLOAT16 / BF16 / INT64 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnReverseSequenceCust

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnReverseSequenceCustGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- input 必须为 3D，seqLengths 必须为 1D，且 `input.shape[0] == seqLengths.shape[0]`
- input 数据类型仅支持 FLOAT32 / FLOAT16 / BF16 / INT64，seqLengths 仅支持 INT64 / INT32
- seqLengths 元素取值应在 [0, max_seq_len] 范围内；越界值由 kernel 防御性截断（大于 max_seq_len 按 max_seq_len 处理），负值等效于 0（不逆序）
- kernel 仅覆写每个 batch 前 seq_len 行，out 需先用 input 数据初始化
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_reverse_sequence.cpp](../examples/aclnn/test_aclnn_reverse_sequence.cpp) 的完整实现。
