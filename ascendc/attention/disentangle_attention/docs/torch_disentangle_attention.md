# torch_disentangle_attention

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |
| Atlas A3  训练/推理系列产品 | √ |
| Atlas A2  训练/推理系列产品 | √ |

## 功能说明

实现 Disentangle Attention 机制的 PyTorch 接口，支持三种位置编码模式：
- c2p：当前到过去的注意力
- p2c：过去到当前的注意力
- c2p|p2c：双向注意力

## 函数原型

```python
ops_rec.ascendc.attention.disentangle_attention(
    query_layer: Tensor,
    key_layer: Tensor,
    value_layer: Tensor,
    pos_key_layer: Tensor,
    pos_query_layer: Tensor,
    relative_pos: Tensor,
    attn_mask: Tensor,
    pos_attr_type: str = "c2p|p2c",
    score_scale: float = 1.0
) -> Tuple[Tensor, Tensor, Tensor]
```

### 参数说明

| 参数名 | 类型 | 描述 | 约束 |
|--------|------|------|------|
| query_layer | Tensor | 查询层，shape [B, N, S, D] | float16, 4D, S=256 |
| key_layer | Tensor | 键层，shape [B, N, S, D] | float16, 4D, 与 query_layer shape 相同 |
| value_layer | Tensor | 值层，shape [B, N, S, D] | float16, 4D, 与 query_layer shape 相同 |
| pos_key_layer | Tensor | 位置键层，shape [2*S, N, D] | float16, 3D |
| pos_query_layer | Tensor | 位置查询层，shape [2*S, N, D] | float16, 3D |
| relative_pos | Tensor | 相对位置编码，shape [1, 1, S, S] 或 [1, S, S] 或 [S, S] | int64, 2D/3D/4D |
| attn_mask | Tensor | 注意力掩码，shape [B, 1, S, S] 或 [B, S, S] | float16, 3D/4D |
| pos_attr_type | str | 位置属性类型，可选 "c2p"、"p2c"、"c2p\|p2c" | 字符串 |
| score_scale | float | 分数缩放因子 | 浮点数，默认 1.0 |

### 返回值

返回三个 Tensor 的元组：

| 返回值 | 描述 | Shape | 数据类型 |
|--------|------|-------|----------|
| outputs | 注意力输出 | [B, N, S, D] | float16 |
| probs | 注意力概率 | [B, N, S, S] | float16 |
| weights | 注意力权重 | [B, N, S, S] | float16 |

## 约束说明

### 数据类型约束

- query_layer、key_layer、value_layer、pos_key_layer、pos_query_layer、attn_mask：必须为 float16
- relative_pos：必须为 int64

### 维度约束

- query_layer、key_layer、value_layer：必须为 4D，且 S (seq_len) 必须为 256
- pos_key_layer、pos_query_layer：必须为 3D，第一维必须为 2*S = 512
- relative_pos：支持 2D/3D/4D，但最后两维必须为 [S, S] = [256, 256]
- attn_mask：支持 3D [B, S, S] 或 4D [B, 1, S, S]

### 数值范围约束

- batch_size (B)：[0, 100]
- head_num (N)：[0, 64]  
- head_dim (D)：[1, 512]，且必须是 16 的倍数
- seq_len (S)：固定为 256

### 设备约束

- 所有输入 Tensor 必须在同一 NPU 设备上
- 该算子仅在 NPU 设备上运行

### 附加约束

- key_layer、value_layer 必须与 query_layer shape 完全相同
- pos_key_layer、pos_query_layer 必须与 pos_key_layer shape 完全相同
- pos_attr_type 必须是 "c2p"、"p2c"、"c2p|p2c" 之一

## 异常说明

当输入参数不满足约束时，会抛出以下异常：

- `RuntimeError: float16 xxx tensor expected but got dtype: xxx` - 数据类型错误
- `RuntimeError: xxx expect X dim but get X` - 维度数量错误
- `RuntimeError: query_layer format must equal value_layer format` - shape 不匹配
- `RuntimeError: pos_attr is invalid must in [c2p, p2c, c2p|p2c]` - 位置属性类型错误
- `RuntimeError: current seq only support 256 but get xxx` - 序列长度不是 256

## 调用示例

### 基础用法

```python
import torch
import ops_rec

torch.npu.set_device(0)

# 准备输入数据
b, n, s, d = 2, 4, 256, 32
query_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
key_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
value_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
pos_key_layer = torch.randn(2*s, n, d, device='npu', dtype=torch.float16)
pos_query_layer = torch.randn(2*s, n, d, device='npu', dtype=torch.float16)
relative_pos = torch.randint(0, 10, (1, 1, s, s), device='npu', dtype=torch.int64)
attn_mask = torch.randn(b, 1, s, s, device='npu', dtype=torch.float16)

# 调用算子
outputs, probs, weights = ops_rec.ascendc.attention.disentangle_attention(
    query_layer, key_layer, value_layer,
    pos_key_layer, pos_query_layer, relative_pos, attn_mask,
    pos_attr_type="c2p|p2c", score_scale=1.0
)

print(f"Output shape: {outputs.shape}")  # [2, 4, 256, 32]
print(f"Probs shape: {probs.shape}")     # [2, 4, 256, 256]
print(f"Weights shape: {weights.shape}") # [2, 4, 256, 256]
```

### 不同位置编码模式

```python
# c2p 模式：当前到过去的注意力
outputs_c2p, _, _ = ops_rec.ascendc.attention.disentangle_attention(
    query_layer, key_layer, value_layer,
    pos_key_layer, pos_query_layer, relative_pos, attn_mask,
    pos_attr_type="c2p", score_scale=1.0
)

# p2c 模式：过去到当前的注意力
outputs_p2c, _, _ = ops_rec.ascendc.attention.disentangle_attention(
    query_layer, key_layer, value_layer,
    pos_key_layer, pos_query_layer, relative_pos, attn_mask,
    pos_attr_type="p2c", score_scale=1.0
)

# c2p|p2c 模式：双向注意力
outputs_both, _, _ = ops_rec.ascendc.attention.disentangle_attention(
    query_layer, key_layer, value_layer,
    pos_key_layer, pos_query_layer, relative_pos, attn_mask,
    pos_attr_type="c2p|p2c", score_scale=1.0
)
```

### torch.compile 图模式

```python
import torch
import ops_rec

# 定义模型
def model(query, key, value, pos_key, pos_query, rel_pos, mask):
    return ops_rec.ascendc.attention.disentangle_attention(
        query, key, value, pos_key, pos_query, rel_pos, mask,
        pos_attr_type="c2p|p2c", score_scale=1.0
    )

# 编译模型
compiled_model = torch.compile(model, dynamic=True, fullgraph=True)

# 调用编译后的模型
outputs, probs, weights = compiled_model(
    query_layer, key_layer, value_layer,
    pos_key_layer, pos_query_layer, relative_pos, attn_mask
)
```

## 更多示例

具体参考 [test_torch_disentangle_attention.py](../examples/torch/test_torch_disentangle_attention.py) 的完整实现。