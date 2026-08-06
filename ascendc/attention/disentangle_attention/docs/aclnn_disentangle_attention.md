# aclnnDisentangleAttention

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

实现 Disentangle Attention 机制，支持三种位置编码模式：
- c2p：当前到过去的注意力
- p2c：过去到当前的注意力  
- c2p|p2c：双向注意力

计算公式：

```text
# 基础注意力权重
attn_weights = matmul(query_layer * score_scale, key_layer^T)

# 位置编码偏置
if "c2p" in pos_attr_type:
    c2p_att = matmul(query_layer, pos_key_layer^T)
    c2p_att = gather(c2p_att, relative_pos_clamped)
    attn_bias += c2p_att

if "p2c" in pos_attr_type:
    p2c_att = matmul(key_layer, pos_query_layer^T)
    p2c_att = gather(p2c_att, relative_pos_clamped)
    attn_bias += p2c_att.transpose(-1, -2)

attn_weights = attn_weights + attn_bias * score_scale + atten_mask
attn_probs = softmax(attn_weights)
attn_outputs = matmul(attn_probs, value_layer)
```

## 函数原型

两段式接口，必须先调用 `aclnnDisentangleAttentionGetWorkspaceSize` 接口获取 workspace 大小和执行器，再调用 `aclnnDisentangleAttention` 接口执行计算。

```c
aclnnStatus aclnnDisentangleAttentionGetWorkspaceSize(
    const aclTensor *query_layer,
    const aclTensor *key_layer,
    const aclTensor *value_layer,
    const aclTensor *pos_key_layer,
    const aclTensor *pos_query_layer,
    const aclTensor *relative_pos,
    const aclTensor *atten_mask,
    int32_t pos_attr_type,
    float score_scale,
    const aclTensor *atten_outputs,
    const aclTensor *atten_probs,
    const aclTensor *atten_weights,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnDisentangleAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnDisentangleAttentionGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|---------|------|---------|---------|
| query_layer（aclTensor） | 输入 | 查询层，shape [B, N, S, D] | FLOAT16 | ND |
| key_layer（aclTensor） | 输入 | 键层，shape [B, N, S, D] | FLOAT16 | ND |
| value_layer（aclTensor） | 输入 | 值层，shape [B, N, S, D] | FLOAT16 | ND |
| pos_key_layer（aclTensor） | 输入 | 位置键层，shape [2*S, N, D] | FLOAT16 | ND |
| pos_query_layer（aclTensor） | 输入 | 位置查询层，shape [2*S, N, D] | FLOAT16 | ND |
| relative_pos（aclTensor） | 输入 | 相对位置，shape [1, 1, S, S] 或 [1, S, S] 或 [S, S] | INT64 | ND |
| atten_mask（aclTensor） | 输入 | 注意力掩码，shape [B, 1, S, S] 或 [B, S, S] | FLOAT16 | ND |
| pos_attr_type（int32_t） | 输入 | 位置属性类型，0=c2p, 1=p2c, 2=c2p|p2c | - | - |
| score_scale（float） | 输入 | 分数缩放因子 | - | - |
| atten_outputs（aclTensor） | 输出 | 注意力输出，shape [B, N, S, D] | FLOAT16 | ND |
| atten_probs（aclTensor） | 输出 | 注意力概率，shape [B, N, S, S] | FLOAT16 | ND |
| atten_weights（aclTensor） | 输出 | 注意力权重，shape [B, N, S, S] | FLOAT16 | ND |
| workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | - |
| executor（aclOpExecutor**） | 输出 | 返回算子执行器 | - | - |

### 返回值

| 返回值 | 错误码 | 描述 |
|--------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入参数是必选输入/输出且为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数数据类型或数据格式不在支持范围内 |

## aclnnDisentangleAttention

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由 `aclnnDisentangleAttentionGetWorkspaceSize` 获取 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 指定执行任务的 Stream |

### 返回值

aclnnStatus：返回状态码，成功返回 `ACLNN_SUCCESS`。

## 约束说明

- 仅支持 FLOAT16 数据类型
- query_layer、key_layer、value_layer 必须为 4D，shape [B, N, S, D]，其中 S 必须为 256
- pos_key_layer、pos_query_layer 必须为 3D，shape [2*S, N, D]
- relative_pos 支持 2D/3D/4D，但最后两维必须为 [S, S]
- atten_mask 支持 3D/4D
- head_dim (D) 必须是 16 的倍数
- batch_size (B) 范围 [0, 100]
- head_num (N) 范围 [0, 64]
- head_dim (D) 范围 [1, 512]
- 该算子需要在 NPU 设备上运行

## 调用示例

具体参考 [test_aclnn_disentangle_attention.cpp](../examples/aclnn/test_aclnn_disentangle_attention.cpp) 的完整实现。