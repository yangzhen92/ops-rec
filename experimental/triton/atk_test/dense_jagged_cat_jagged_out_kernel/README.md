# dense_jagged_cat_jagged_out_kernel

## 来源

`dense_jagged_cat_jagged_out_kernel` 源自 PyTorch FBGEMM：

<https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_dense_jagged_cat_jagged_out.py>

## 算子功能

算子将每个 batch 的一个 Dense 标量放到对应 Jagged 序列之前。设
`b_offsets=[0, s_0, ..., s_B]`，第 i 段输出为：

```text
c[c_offsets[i]] = a[i]
c[c_offsets[i] + 1 : c_offsets[i + 1]] = b[s_i : s_(i+1)]
c_offsets[i] = s_i + i
```

例如 `a=[10, 20]`、`b=[1, 2, 3]`、`b_offsets=[0, 2, 3]` 时，输出
`c=[10, 1, 2, 20, 3]`，`c_offsets=[0, 3, 5]`。

## 输入与输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `a_ptr` | `(B,)` | fp32 / fp16 / bf16 | 每个 batch 的 Dense 标量 |
| `b_ptr` | `(T,)` | 与 `a_ptr` 一致 | 所有 Jagged 段连续存储的值 |
| `b_offsets_ptr` | `(B + 1,)` | int32 | `b_ptr` 中每段的起止偏移 |
| `max_seq_len` | 标量 | int | 单段最大处理长度 |
| `c_ptr` | `(T + B,)` | 与 `a_ptr` 一致 | 在每段开头插入 Dense 标量后的结果 |

ATK API 根据 B 和 T 确定性地生成各段长度、offsets 和有效 `max_seq_len`，使 GPU 与
NPU 使用完全相同的 Jagged 布局。`c_offsets_ptr` 是 Kernel 的辅助输出；当前用例返回并
比较 `c_ptr`。

## 关键约束

- `a_ptr`、`b_ptr`、`b_offsets_ptr` 均为一维连续张量。
- `a_ptr` 与 `b_ptr` 的 dtype 相同，`b_offsets_ptr` 为 int32。
- `B = a_ptr.numel()`，`b_offsets_ptr.numel() = B + 1`。
- offsets 单调不减，首项为 0，末项为 `T = b_ptr.numel()`。
- 每段长度不超过有效 `max_seq_len`，空段是合法输入。

## 使用样例

```python
import torch

from triton_dense_jagged_cat_jagged_out import dense_jagged_cat_jagged_out

device = "npu"
a = torch.tensor([10, 20], dtype=torch.float32, device=device)
b = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
offsets = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
c, c_offsets = dense_jagged_cat_jagged_out(a, b, offsets, max_seq_len=2)
print(c, c_offsets)
```

## 目录文件

```text
dense_jagged_cat_jagged_out_kernel/
├── README.md
├── gpu_atk_test/
│   ├── triton_dense_jagged_cat_jagged_out.py
│   └── dense_jagged_cat_jagged_out_kernel_api.py
└── npu_atk_test/
    ├── triton_dense_jagged_cat_jagged_out.py
    ├── dense_jagged_cat_jagged_out_kernel_api.py
    ├── dense_jagged_cat_jagged_out_kernel.yaml
    ├── generate_dense_jagged_cat_jagged_out_kernel.py
    └── dense_jagged_cat_jagged_out_kernel_performance.json
```

## ATK 测试
 
测试在 NPU / GPU 两端协同进行：
 
- `npu_atk_test/` 部署到 NPU 端；`gpu_atk_test/` 部署到 GPU 端。
- GPU 端启动 ATK server，NPU 端作为 ATK 客户端拉起精度/性能测试。
 
### 远端 GPU 服务器起监听命令
 
```bash
atk server --devices 0 --plugin_path /path/to/api文件所在文件夹
```
 
### 3.1 生成测试数据
 
```bash
atk case -f dense_jagged_cat_jagged_out_kernel.yaml \
    -p generate_dense_jagged_cat_jagged_out_kernel.py
```
 
### 3.2 精度对比
 
```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c all_dense_jagged_cat_jagged_out_kernel.json --task accuracy \
    -p dense_jagged_cat_jagged_out_kernel_api.py
```
 
注：如需保存精度对比数据，需要在命令后加 `--save_data profile`。
 
### 3.3 性能对比
 
```bash
atk node --backend npu --devices 0 node --backend gpu -h {gpu端ip} -p {端口} --devices 0 \
    task -c dense_jagged_cat_jagged_out_kernel_performance.json --task performance_device \
    -p dense_jagged_cat_jagged_out_kernel_api.py
```
 
注：如需保存性能对比数据，需要在命令后加 `--save_data profile`。
