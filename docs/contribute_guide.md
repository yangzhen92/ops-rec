# 贡献指南

## 上库准备

- 已了解目录结构和两种算子模式（见本文档后续章节）
- 本地已通过 `bash build.sh --socs <variant> --install -j<N>` 完成编译验证
- 算子示例已通过 `bash run_examples.sh --ops <vendor>` 验证正确性

---

## 概述

### 目录结构约定

```text
ascendc/                         ← C++ 算子源码（CMake 编译单元）
  <group>/                           ← 功能分组（activation / attention / embedding ...）
    <vendor>/                        ← vendor 名，也是 CANN package 名
      op_host/                        ← OpDef 注册 + Tiling + InferShape
      op_kernel/                      ← AscendC kernel 实现
      torch_extension/                ← TORCH_LIBRARY 注册（可选）
      examples/
        aclnn/                        ← aclnn C 接口调用示例
        torch/                        ← torch 接口调用示例
      docs/                           ← 算子 API 文档
      test/                           ← 单元测试
      CMakeLists.txt                  ← 聚合编译
```

### 两种算子模式

#### 模式一：单算子 vendor（如 ln_mul）

一个 vendor 目录只包含一个算子，通过 `activation/__init__.py` 的 `__getattr__` 直接代理到 `torch.ops.ops_rec`，无需每算子写 Python 封装文件。

目录结构：

```text
ascendc/activation/ln_mul/
  op_host/ln_mul.cpp               → OP_ADD(LnMul);
  op_host/ln_mul_tiling.h
  op_host/CMakeLists.txt
  op_kernel/ln_mul_kernel.h
  op_kernel/CMakeLists.txt
  torch_extension/ln_mul.cpp       → TORCH_LIBRARY(ops_rec, m) { m.def("ln_mul", ...); }

ops_rec/ascendc/activation/
  __init__.py                       ← __getattr__ 代理到 torch.ops.ops_rec
```

调用方式：

```python
import ops_rec
ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)
```

`activation/__init__.py` 的 `__getattr__` 会从 `torch.ops.ops_rec` 查找算子名。这是默认模式，适合大多数算子。

#### 模式二：多算子 vendor（如 hstu_v1）

一个 vendor 目录下有多个算子，共享 tiling/kernel 代码。每个算子在 C++ 侧各自 `OP_ADD`，Python 侧通过 `vendor/__init__.py` 手动封装或统一代理到 `torch.ops.ops_rec`。

目录结构：

```text
ascendc/attention/hstu_v1/
  op_host/
    hstu_dense_forward.cpp          → OP_ADD(HstuDenseForward);
    hstu_jagged_forward.cpp         → OP_ADD(HstuJaggedForward);
    hstu_paged_forward.cpp          → OP_ADD(HstuPagedForward);
    tiling_*.cpp/h
  op_kernel/
    hstu_dense_forward.cpp
    hstu_jagged_forward.cpp
    hstu_paged_forward.cpp
  torch_extension/
    hstu_v1.cpp                     → TORCH_LIBRARY(ops_rec, m) {
                                      m.def("hstu_dense", ...);
                                      m.def("hstu_jagged", ...);
                                      m.def("hstu_paged", ...);
                                    }
  CMakeLists.txt                    → vendor 名 hstu_v1

ops_rec/ascendc/attention/
  hstu_v1/
    __init__.py                     ← __getattr__ 代理到 torch.ops.ops_rec
```

调用方式：

```python
import ops_rec
ops_rec.ascendc.attention.hstu_v1.hstu_dense(x)
ops_rec.ascendc.attention.hstu_v1.hstu_jagged(x)
```

`hstu_v1/__init__.py` 模板：

```python
import torch

_OP_NAMESPACE = torch.ops.ops_rec

def __getattr__(name):
    if hasattr(_OP_NAMESPACE, name):
        return getattr(_OP_NAMESPACE, name)
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'"
    )

def __dir__():
    own_attrs = set(globals().keys())
    op_attrs = {k for k in dir(_OP_NAMESPACE) if not k.startswith("_")}
    return sorted(own_attrs | op_attrs)
```

---

## AscendC 算子迁移

### 迁移步骤

#### 1. 创建目录

```bash
# 单算子
mkdir -p ascendc/<group>/<vendor>/op_host
mkdir -p ascendc/<group>/<vendor>/op_kernel
mkdir -p ascendc/<group>/<vendor>/torch_extension
mkdir -p ascendc/<group>/<vendor>/examples/aclnn
mkdir -p ascendc/<group>/<vendor>/examples/torch
mkdir -p ascendc/<group>/<vendor>/docs
mkdir -p ascendc/<group>/<vendor>/test/pytest
```

#### 2. 复制 C++ 源码

将 `op_host/` 和 `op_kernel/` 下的源码放入对应目录。

#### 3. 编写 CMakeLists.txt（参考 ln_mul）

- `ascendc/<group>/<vendor>/CMakeLists.txt`：聚合子目录，设置 vendor 名
- `op_host/CMakeLists.txt`：`npu_op_code_gen` + `npu_op_library`
- `op_kernel/CMakeLists.txt`：`npu_op_kernel_library`

#### 4. 注册到 SOC 列表

`cmake/soc_ops_config.cmake` 中将 vendor 名加入对应 SOC 列表：

```cmake
set(OPS_REC_OPS_ATLAS_950 ln_mul hstu_v1 <new_vendor>)
```

#### 5. 编写 torch_extension（可选）

在 `torch_extension/` 下使用 `TORCH_LIBRARY(ops_rec, ...)` 注册算子，编译时会自动收集所有 `torch_extension/*.cpp` 并链接到统一的 `.so`。

#### 6. 编写示例

示例文件命名必须遵循 `test_aclnn_<vendor>*.cpp` 和 `test_torch_<vendor>*.py` 格式，`run_examples.sh` 通过 `find` 自动发现：

```bash
examples/aclnn/test_aclnn_<vendor>.cpp           # aclnn C 接口示例
examples/aclnn/test_aclnn_<vendor>_<case>.cpp    # 多个 aclnn 示例时加后缀
examples/torch/test_torch_<vendor>.py             # torch 接口示例
examples/torch/test_torch_<vendor>_<case>.py      # 多个 torch 示例时加后缀
```

运行示例：

```bash
bash run_examples.sh --type torch --ops <vendor>
bash run_examples.sh --type aclnn --ops <vendor>
bash run_examples.sh --type all --ops <vendor>   # aclnn + torch
```

#### 7. 编写文档

每个算子应包含：

```text
ascendc/<group>/<vendor>/
  README.md                     ← 算子使用入口（产品支持、功能、参数、约束）
  docs/
    aclnn_<vendor>.md           ← aclnn C 接口 API 参考
    torch_<vendor>.md           ← torch 接口 API 参考（ops_rec.ascendc.<group>.<vendor>）
```

参考 `ascendc/activation/ln_mul/` 下的文档模板。

#### 8. 编写测试

在 `test/pytest/` 下编写 pytest 用例：

```text
ascendc/<group>/<vendor>/test/pytest/
  pytest.ini                    ← 测试标记配置
  test_<vendor>.py              ← 测试用例
```

#### 9. 调用验证

```bash
# 编译安装
bash build.sh --socs atlas_950 --install -j16

# 运行示例验证
bash run_examples.sh --type torch --ops <vendor>

# Python 直接调用
python3 -c "
import ops_rec
ops_rec.ascendc.<group>.<vendor>(...)
"

# 运行测试
cd ascendc/<group>/<vendor>/test/pytest && pytest test_*.py -x -v
```

---

## Triton 算子迁移

### 目录结构

```text
experimental/triton/              ← 实验性 Triton 算子
  <group>/                            ← 功能分组
    __init__.py
    triton_<feature>.py               ← Triton kernel 实现

ops_rec/experimental/triton/      ← Python 包（构建时自动同步）
  <group>/
    __init__.py
    triton_<feature>.py
```

### 迁移步骤

#### 1. 创建源码目录

```bash
mkdir -p experimental/triton/<group>
```

#### 2. 编写 Triton kernel

在 `experimental/triton/<group>/` 下创建算子实现文件：

```text
experimental/triton/<group>/
  __init__.py                     ← 自动发现子模块（pkgutil.iter_modules）
  triton_<feature>.py             ← Triton kernel 实现
```

#### 3. 构建时自动同步

`build.sh` 的 `build_single_soc()` 函数会自动将 `experimental/triton/` 下的 `.py` 文件同步到 `ops_rec/experimental/triton/`（排除 `test/` 目录），构建完成后自动删除临时文件，无需手动操作。

#### 4. 调用验证

```bash
# 构建安装
bash build.sh --socs atlas_950 --install -j16

# 导入验证
python3 -c "
from ops_rec.experimental.triton.<group> import <module>
<trigger_kernel_function>(...)
"
```

---

## 编码规范

### C++

- 遵循 C++17 标准
- 遵循已有文件的代码风格（缩进、命名、注释）
- AscendC kernel 代码使用大写驼峰命名类名
- OpDef 注册使用 `ge::graphStatus` 作为返回值类型
- SOC 编译宏使用 `SUPPORT_950` / `SUPPORT_A2` / `SUPPORT_A3`（通过 CMake 变量 `OPS_REC_SOC_CXX_DEFS` 统一传递）

### CMake

- 每个算子包含根 CMakeLists.txt + `op_host/CMakeLists.txt` + `op_kernel/CMakeLists.txt`
- 所有 SOC 编译宏由根 CMakeLists.txt 的 `OPS_REC_SOC_CXX_DEFS` 统一传递，算子 CMakeLists 不自行定义

### Python

- 遵循 PEP 8
- 单算子 vendor：`activation/__init__.py` 通过 `__getattr__` 直接代理到 `torch.ops.ops_rec`，无需 Python 封装文件
- 多算子 vendor：在 `vendor/__init__.py` 中通过 `__getattr__` 代理到 `torch.ops.ops_rec`

---

## 分支管理

| 分支 | 用途 |
|---|---|
| `main` | 稳定发布分支 |
| `dev` | 开发分支 |
| `feature/<name>` | 功能分支，开发完成后合并到 `dev` |
| `fix/<name>` | 修复分支 |

## PR 提交

### 提交前检查

- [ ] 代码编译通过（目标 SOC）
- [ ] 示例运行正确
- [ ] 测试用例通过
- [ ] 代码风格与仓库一致
- [ ] 已同步上游分支

### PR 模板

```markdown
## 描述

[简要说明变更内容]

## 测试验证

- [ ] 本地编译通过（atlas_950 / atlas_a2 / atlas_a3）
- [ ] 示例通过
- [ ] 测试通过

## 破坏性变更

[如有，列出不兼容的变更]
```

## 许可证

提交代码即表示您同意本仓库的 Apache License 2.0 许可条款。
