# disentangle_attention

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR&950DT 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

实现 Disentangle Attention 机制，支持三种位置编码模式：

- c2p：当前到过去的注意力
- p2c：过去到当前的注意力
- c2p+p2c：双向注意力

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

对应接口：

- aclnn 接口：`aclnnDisentangleAttention`
- torch 接口：`ops_rec.ascendc.attention.disentangle_attention`

## 算子实现原理

算子采用 AscendC 进行开发，主要计算步骤如下：

1. **输入准备**：接收 query_layer、key_layer、value_layer、pos_key_layer、pos_query_layer、relative_pos、atten_mask 等 7 个输入
2. **位置编码计算**：根据 pos_attr_type 分别计算 c2p 和 p2c 位置注意力
3. **矩阵乘法**：并行执行 QK 矩阵乘法、位置编码矩阵乘法、SV 矩阵乘法
4. **注意力权重计算**：合并基础注意力和位置编码偏置，加上掩码
5. **Softmax 归一化**：对注意力权重进行 softmax 操作
6. **输出计算**：使用归一化的注意力概率对 value_layer 加权求和

算子针对序列长度 S=256 进行优化，支持多核并行计算，使用流水线技术提高计算效率。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
|--------|---------|---------|---------|---------|-------|------|
| query_layer | 输入 | Tensor | FLOAT16 | ND | [B, N, S, D] | 查询层，S 必须为 256 |
| key_layer | 输入 | Tensor | FLOAT16 | ND | [B, N, S, D] | 键层 |
| value_layer | 输入 | Tensor | FLOAT16 | ND | [B, N, S, D] | 值层 |
| pos_key_layer | 输入 | Tensor | FLOAT16 | ND | [2*S, N, D] | 位置键层 |
| pos_query_layer | 输入 | Tensor | FLOAT16 | ND | [2*S, N, D] | 位置查询层 |
| relative_pos | 输入 | Tensor | INT64 | ND | [1, 1, S, S] 或 [1, S, S] 或 [S, S] | 相对位置编码 |
| atten_mask | 输入 | Tensor | FLOAT16 | ND | [B, 1, S, S] 或 [B, S, S] | 注意力掩码 |
| pos_attr_type | 输入 | str | - | - | - | 位置属性类型：c2p/p2c/c2p\|p2c |
| score_scale | 输入 | float | - | - | - | 分数缩放因子，默认 1.0 |
| atten_outputs | 输出 | Tensor | FLOAT16 | ND | [B, N, S, D] | 注意力输出 |
| atten_probs | 输出 | Tensor | FLOAT16 | ND | [B, N, S, S] | 注意力概率 |
| atten_weights | 输出 | Tensor | FLOAT16 | ND | [B, N, S, S] | 注意力权重 |

## 约束说明

### 数据类型约束

- query_layer、key_layer、value_layer、pos_key_layer、pos_query_layer、attn_mask：必须为 float16
- relative_pos：必须为 int64

### 维度约束

- query_layer、key_layer、value_layer：必须为 4D，shape [B, N, S, D]，其中 S 必须为 256
- pos_key_layer、pos_query_layer：必须为 3D，shape [2*S, N, D]
- relative_pos：支持 2D/3D/4D，但最后两维必须为 [S, S]
- attn_mask：支持 3D [B, S, S] 或 4D [B, 1, S, S]

### 数值范围约束

- batch_size (B)：[0, 100]
- head_num (N)：[0, 64]
- head_dim (D)：[1, 512]，且必须是 16 的倍数
- seq_len (S)：固定为 256

### 设备约束

- 所有输入 Tensor 必须在同一 NPU 设备上
- 该算子仅在 NPU 设备上运行

> 通过 torch 接口调用时，扩展层还会额外校验：输入 tensor 必须已定义且非空、所有输入 tensor 必须位于同一 NPU 设备、shape 约束等。详见 [disentangle_attention 接口文档](./docs/torch_disentangle_attention.md)。

## 调用示例

- [aclnn 调用示例](./examples/aclnn/test_aclnn_disentangle_attention.cpp)
- [torch 调用示例](./examples/torch/test_torch_disentangle_attention.py)
- [aclnn 接口文档](./docs/aclnn_disentangle_attention.md)
- [torch 接口文档](./docs/torch_disentangle_attention.md)
