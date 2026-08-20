# create_q2k_block_sparse_from_func

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas 950 系列产品 | √ |

## 功能说明

Q2K（正向）块稀疏张量生成：根据函数编码张量 `func_tensor`，为每个 (b, h, q_block)
并行计算其覆盖的 kv_block 列表，输出 mask 块数、full 块数与块索引表，供
`LinearBlockSparseTensors` / FlexAttention block-mask 使用。

- 每个 (q_block, kv_block) 块按 `func_tensor` 中的合法 kv 区间分类为 FULL / MASK / EMPTY
- `full_block_cnt` 记录 FULL 块数，`mask_block_cnt` 记录 MASK 块数
- 统一block_idx布局说明：每行内FULL块id从左往右写，MASK块id从右往左写（其余位置为-1），仅应读取mask/full计数对应的有效位置。
- `check_q_boundary` 为 true 时，部分 q_block（不足 q_block_size）不能判为 FULL

`func_tensor` 布局为 [B, H, func_q_len, n_func]，`n_func` 为偶数，每个 q token 的合法
kv 区间为 `[F_{2i}, F_{2i+1})` 的并集。该算子为 Q2K 正向路径，对应的反向路径见
`create_k2q_block_sparse_from_func`（输出维度按 kv_block 展开）。

## 接口原型

```python
ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
    func_tensor: Tensor,
    q_len: int,
    kv_len: int,
    q_block_size: int,
    kv_block_size: int,
    check_q_boundary: bool,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | shape | 说明 |
| -------- | --------- | --------- | --------- | --------- | ------- | ------ |
| func_tensor | 输入 | Tensor | INT32 | ND | [B, H, func_q_len, n_func] | 函数编码张量，n_func 为偶数 |
| q_len | 输入 | int | - | - | - | Q 序列长度 |
| kv_len | 输入 | int | - | - | - | KV 序列长度 |
| q_block_size | 输入 | int | - | - | - | Q 块大小 |
| kv_block_size | 输入 | int | - | - | - | KV 块大小 |
| check_q_boundary | 输入 | bool | - | - | - | 部分 q_block 是否不允许 FULL（q_boundary 检查） |
| debug | 输入 | bool | - | - | - | 调试用，true 时输出初始化为 -1 |
| mask_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks] | 每个 q_block 的 mask 块数 |
| full_block_cnt | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks] | 每个 q_block 的 full 块数 |
| block_idx | 输出 | Tensor | INT32 | ND | [B, H, num_q_blocks, num_kv_blocks] | 块索引表 |

> num_q_blocks = ceil(q_len / q_block_size)，num_kv_blocks = ceil(kv_len / kv_block_size)。

## 约束说明

- 仅支持 INT32 数据类型
- func_tensor 必须为 4D，n_func 必须为偶数（如 2、4、6、8、...）
- q_len、kv_len、q_block_size、kv_block_size 必须大于 0
- q_block_size、kv_block_size 必须是32的倍数
- block_idx 每行尾端可能被过度分配（未初始化），只应读取前full_block_cnt个FULL块及后mask_block_cnt个MASK块的有效数据
- 该算子需要在 NPU 设备上运行

## 安装方式

```bash
bash build.sh --install -j16
```

## 调用示例

完整调用示例参考 [test_torch_block_sparse_func_create_q2k_block_sparse_from_func.py](../examples/torch/test_torch_block_sparse_func_create_q2k_block_sparse_from_func.py)。
