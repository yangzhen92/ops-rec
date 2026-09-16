# 贡献指南

## 概述

本指南介绍如何向 ops-rec 仓库新增 AscendC 算子或 Triton 算子，包括目录创建、代码组织、SOC 注册、构建验证与 PR 提交流程。

### 开发流程

新增算子时，按以下顺序推进：

1. [环境部署](../README.md)：克隆仓库、安装构建依赖、验证 CANN / NPU 环境。
2. [目录约定](#目录约定)：确定算子归属到哪个 `<group>/<vendor>` 下。
3. [创建目录](#创建目录)：按约定创建算子目录骨架。
4. [实现 OpDef + Tiling + InferShape](#实现-opdef--tiling--infershape)：编写 `op_host/` 下的 host 侧代码。
5. [实现 Kernel](#实现-kernel)：编写 `op_kernel/` 下的 AscendC kernel。
6. [实现 torch_extension](#实现-torch_extension)：通过 `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL` 注册算子。
7. [注册 SOC](#注册-soc)：将 vendor 名加入 `cmake/soc_ops_config.cmake`。
8. [编写示例与文档](#编写示例与文档)：编写 aclnn / torch 示例、API 文档与 README。
9. [编写测试](#编写测试)：编写 pytest 用例。
10. [构建与验证](#构建与验证)：编译目标 SOC、安装 whl、跑示例与测试。

### 上库准备 Checklist

完成以上流程并提交 PR 前，请确认：

- [ ] 本地编译通过：`bash build.sh --socs <variant> --install -j<N>`
- [ ] 示例运行正确：`bash run_examples.sh --ops <vendor>`
- [ ] pytest 用例通过：`cd ascendc/<group>/<vendor>/test/pytest && pytest test_*.py -x -v`
- [ ] 文档齐全：`README.md` + `docs/aclnn_<op_name>.md` + `docs/torch_<op_name>.md`
- [ ] 代码风格与仓库一致（参考 `.clang-format` / `.clang-tidy` / `.pre-commit-config.yaml`）

---

## 目录约定

ops-rec 的算子按 `<group>/<vendor>` 组织：`group` 是功能分组（如 `activation` / `attention`），`vendor` 是算子注册单元，vendor 名即 CANN package 名。

---

## 创建目录

按上文目录约定创建算子骨架：

### Vendor 骨架

```bash
VENDOR=<vendor_name>           # 小写下划线，如 ln_mul
GROUP=<group_name>             # 功能分组，如 activation

mkdir -p ascendc/${GROUP}/${VENDOR}/op_host
mkdir -p ascendc/${GROUP}/${VENDOR}/op_kernel
mkdir -p ascendc/${GROUP}/${VENDOR}/torch_extension
mkdir -p ascendc/${GROUP}/${VENDOR}/examples/aclnn
mkdir -p ascendc/${GROUP}/${VENDOR}/examples/torch
mkdir -p ascendc/${GROUP}/${VENDOR}/docs
mkdir -p ascendc/${GROUP}/${VENDOR}/test/pytest
```

最终目录结构：

```text
ascendc/<group>/<vendor>/
├── CMakeLists.txt                  # 聚合编译 + vendor 名设置
├── op_host/
│   ├── CMakeLists.txt              # npu_op_code_gen + npu_op_library
│   ├── <op_name>.cpp               # TilingFunc + InferShape + OpDef 注册
│   └── <op_name>_tiling.h          # TilingData 定义
├── op_kernel/
│   ├── CMakeLists.txt              # npu_op_kernel_library
│   ├── <op_name>.cpp               # __global__ __aicore__ kernel 入口
│   └── <op_name>_kernel.h          # Kernel 类实现
├── torch_extension/
│   └── <op_name>_plugin.cpp        # TORCH_LIBRARY 注册 + NPU 实现
├── examples/
│   ├── aclnn/test_aclnn_<op_name>.cpp
│   └── torch/test_torch_<op_name>.py
├── docs/
│   ├── aclnn_<op_name>.md
│   └── torch_<op_name>.md
├── test/pytest/
│   ├── pytest.ini
│   └── test_<op_name>.py
└── README.md
```

> 注册多个算子时，可在 `op_host/`、`op_kernel/`、`torch_extension/` 下任选一种组织方式：
>
> - **按算子名拆分**：每个算子独立文件（如 `op_a.cpp` / `op_b.cpp`），便于维护与代码评审，适合算子较多或彼此差异大的场景；
> - **共用 vendor 名文件**：在 `<vendor>.cpp` 内定义多个 OpDef 类与 kernel 函数，适合算子少且关联紧密的场景。
>
> 编译系统会按目录自动收集 `.cpp`，并链接到同一个 `ops_rec_atlas_*.so`；同 vendor 下所有算子共用 tiling target，多算子时需用不同 `TilingKey` 区分。

---

## 实现 OpDef + Tiling + InferShape

在 `op_host/<op_name>.cpp` 中实现 `TilingFunc`、`InferShape` 与 `OpDef`：

```cpp
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 计算核数、workspace、每核处理行数等
    ...
    context->SetBlockDim(coreNum);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* in_shape = context->GetInputShape(0);
    *context->GetOutputShape(0) = *in_shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MyOp : public OpDef {
public:
    explicit MyOp(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");
        this->AICore().SetTiling(optiling::TilingFunc);

        // 按 SOC 添加配置：ascend910b (A2) / ascend910_93 (A3) / ascend950 / ascend310p
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
    }
};
OP_ADD(MyOp);
}  // namespace ops
```

`op_host/<op_name>_tiling.h` 定义 TilingData 结构体：

```cpp
namespace optiling {
BEGIN_TILING_DATA_DEF(MyOpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, aLength);
TILING_DATA_FIELD_DEF(uint32_t, rLength);
...
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)
}  // namespace optiling
```

---

## 实现 Kernel

`op_kernel/<op_name>.cpp` 中定义 AICore 入口：

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void my_op(GM_ADDR xGm, GM_ADDR yGm,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    MyOpKernel::Args args{xGm, yGm, workspace, tiling};
    if (TILING_KEY_IS(1)) {
        MyOpKernel::KernelImpl op;
        op.Init(args);
        op.Process();
    }
}
```

`op_kernel/<op_name>_kernel.h` 中实现 Kernel 类（`CopyIn` / `Compute` / `CopyOut` 三阶段），参考 `ln_mul_kernel.h`。

### SOC 架构特化代码

如同一算子在Ascend 950PR&950DT系列产品与Atlas A2/A3 系列产品上实现差异较大，可将特化代码放入子目录：

```text
op_host/atlas950/<op_name>_tiling.cpp    # Ascend 950PR&950DT系列产品 SOC 编译
op_host/atlasA2/<op_name>_tiling.cpp     # 仅 Atlas A2/A3 系列产品 SOC 编译
op_kernel/atlas950/<op_name>.cpp
op_kernel/atlasA2/<op_name>.cpp
```

根 `CMakeLists.txt` 已通过 `cmake/ops_rec_arch.cmake` 提供的 `ops_rec_collect_arch_sources(<var>)` 函数自动按 `OPS_REC_ARCH_DIR` 收集。算子级 CMakeLists 中调用：

```cmake
ops_rec_collect_arch_sources(ops_srcs)
```

无需自行判断 SOC。

---

## 实现 torch_extension

所有算子统一通过 `TORCH_LIBRARY_FRAGMENT` + `TORCH_LIBRARY_IMPL` 注册到 `ops_rec` 命名空间，编译时由根 `CMakeLists.txt` 自动收集所有 `ascendc/*/*/torch_extension/*.cpp` 链接到统一的 `ops_rec_atlas_<variant>.so`。

### 示例

`torch_extension/<op_name>_plugin.cpp`：

```cpp
// 声明本插件依赖的 vendor（a c l nn 接口所在 op_api 库）：
// 单个 vendor ：#define OPS_REC_CUST_VENDORS "ln_mul"
// 多个 vendor ：#define OPS_REC_CUST_VENDORS "hstu_forward_v1,hstu_backward_v1"（逗号分隔）
// 必须在 #include "pytorch_npu_helper.hpp" 之前定义，否则 EXEC_NPU_CMD 展开处宏未定义会编译报错。
#define OPS_REC_CUST_VENDORS "my_vendor"

#include <torch/library.h>
#include "common_utils.h"
#include "pytorch_npu_helper.hpp"

at::Tensor my_op_impl_npu(const at::Tensor& x)
{
    check_tensor_non_empty(x, "x");
    std::vector<at::Tensor> tensors = {x};
    std::vector<std::string> names = {"x"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(x.dim() == 2, "The x should be 2D");

    auto x_conti = x.contiguous();
    at::Tensor y = at::empty_like(x_conti);
    // 调用 aclnn 接口（自动处理 workspaceSize / executor / stream）
    EXEC_NPU_CMD(aclnnMyOp, x_conti, y);
    return y;
}

// 注册 schema
TORCH_LIBRARY_FRAGMENT(ops_rec, m)
{
    m.def("my_op(Tensor x) -> Tensor");
}

// 注册 NPU 实现（PrivateUse1 = Ascend NPU 后端标识）
TORCH_LIBRARY_IMPL(ops_rec, PrivateUse1, m)
{
    m.impl("my_op", &my_op_impl_npu);
}

// 可选：注册 Meta 实现，用于 torch.compile 的 shape 推断
TORCH_LIBRARY_IMPL(ops_rec, Meta, m)
{
    m.impl("my_op", &my_op_meta);
}
```

关键点：

- **`TORCH_LIBRARY_FRAGMENT`**：声明算子的 schema（入参类型 + 返回值）。
- **`TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)`**：注册 NPU 设备实现。
- **`TORCH_LIBRARY_IMPL(..., Meta, ...)`**（可选）：注册 Meta 实现，支持 `torch.compile` shape 推断。
- **`EXEC_NPU_CMD(aclnnXxx, ...)`**：定义在 `ascendc/common/torch_plugin/pytorch_npu_helper.hpp`，封装 workspace 申请、执行器管理、流绑定等。它通过宏 `OPS_REC_CUST_VENDORS` 定位 `aclnnXxx` 符号所在的 op_api 库（vendor）。
- **`OPS_REC_CUST_VENDORS`（必填）**：每个插件**必须**在包含 `pytorch_npu_helper.hpp` **之前** `#define OPS_REC_CUST_VENDORS "..."`，值是该插件 aclnn 接口所在的一个或多个 vendor（逗号分隔，如 `"hstu_forward_v1,hstu_backward_v1"`）。运行时会按逗号拆分、依次在各 vendor 中查找 aclnn 符号。该宏未定义会直接编译报错，多个 vendor 只能通过逗号分隔声明，不再由 CMake 目录结构推断。

### Python 代理

`<group>/__init__.py`（如 `ascendc/activation/__init__.py`）通过 `__getattr__` 代理到 `torch.ops.ops_rec`，无需为每个算子单独写 Python 文件。

---

## 注册 SOC

编辑 `cmake/soc_ops_config.cmake`，将 vendor 名加入对应 SOC 的列表：

```cmake
set(OPS_REC_OPS_ATLAS_950  ln_mul <new_vendor>)
set(OPS_REC_OPS_ATLAS_A3   ln_mul <new_vendor>)
set(OPS_REC_OPS_ATLAS_A2   ln_mul <new_vendor>)
```

根 `CMakeLists.txt` 会自动按该列表筛选算子参与编译，并据此生成 `vendors/config.ini` 的 `load_priority`。

---

## 编写示例与文档

### 示例命名规范

示例文件必须遵循 `test_<kind>_<op_name>*.cpp` / `test_<kind>_<op_name>*.py` 格式，`run_examples.sh` 通过 `find` 自动发现：

```bash
examples/aclnn/test_aclnn_<op_name>.cpp            # aclnn C 接口示例
examples/aclnn/test_aclnn_<op_name>_<case>.cpp     # 多个 aclnn 示例时加后缀
examples/torch/test_torch_<op_name>.py              # torch 接口示例
examples/torch/test_torch_<op_name>_<case>.py       # 多个 torch 示例时加后缀
```

### 运行示例

```bash
bash build.sh --socs atlas_950 --install -j16    # 先安装

bash run_examples.sh --type torch --ops <vendor>
bash run_examples.sh --type aclnn --ops <vendor>
bash run_examples.sh --type all   --ops <vendor>
```

### 文档规范

每个算子应包含：

| 文档 | 位置 | 内容 |
|---|---|---|
| 算子 README | `ascendc/<group>/<vendor>/README.md` | 产品支持、功能、参数、约束、调用示例入口 |
| aclnn API 参考 | `ascendc/<group>/<vendor>/docs/aclnn_<op_name>.md` | `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 签名、参数表、错误码 |
| torch API 参考 | `ascendc/<group>/<vendor>/docs/torch_<op_name>.md` | `ops_rec.ascendc.<group>.<op_name>` 签名、参数表、torch 扩展层额外校验 |

参考 `ascendc/activation/ln_mul/` 下的文档模板。

---

## 编写测试

在 `test/pytest/` 下编写 pytest 用例：

```text
ascendc/<group>/<vendor>/test/pytest/
├── pytest.ini                    # 测试标记配置（ci / graph）
└── test_<op_name>.py             # 用例主体
```

用例示例参考 `ascendc/activation/ln_mul/test/pytest/test_ln_mul.py`，通常覆盖：

- 多种 shape 组合的 eager 模式（`@pytest.mark.parametrize`）。
- `@pytest.mark.graph` 修饰的 `torch.compile` 图模式用例。
- `@pytest.mark.ci` 修饰的固定 shape 冒烟用例（CI 必跑）。

运行：

```bash
cd ascendc/<group>/<vendor>/test/pytest && pytest test_*.py -x -v
```

---

## 构建与验证

完整流程：

```bash
# 1. 编译安装（指定目标 SOC）
bash build.sh --socs atlas_950 --install -j16

# 2. 运行示例
bash run_examples.sh --type torch --ops <vendor>
bash run_examples.sh --type aclnn --ops <vendor>

# 3. Python 直接调用
python3 -c "
import ops_rec
ops_rec.ascendc.<group>.<op_name>(...)
"

# 4. 跑测试
cd ascendc/<group>/<vendor>/test/pytest && pytest test_*.py -x -v
```

---

## 编码规范

### C++

- 遵循 C++17 标准。
- 遵循已有文件的代码风格（缩进、命名、注释）。
- AscendC kernel 代码使用大写驼峰命名类名。
- OpDef 注册使用 `ge::graphStatus` 作为返回值类型。
- SOC 编译宏使用 `SUPPORT_950` / `SUPPORT_A2` / `SUPPORT_A3`，由根 CMakeLists 的 `OPS_REC_SOC_CXX_DEFS` 统一传递。

### CMake

- 每个算子包含根 `CMakeLists.txt` + `op_host/CMakeLists.txt` + `op_kernel/CMakeLists.txt`。
- 所有 SOC 编译宏由根 `CMakeLists.txt` 统一传递，算子级 CMakeLists 不自行定义。

### Python

- 遵循 PEP 8。
- `<group>/__init__.py` 通过 `__getattr__` 直接代理到 `torch.ops.ops_rec`。

---

## 贡献流程

### 分支约定

| 分支 | 用途 |
|---|---|
| `master` | 稳定发布分支 |
| `dev` | 开发分支 |
| `feature/<name>` | 功能分支，合并回 `dev` |
| `fix/<name>` | 修复分支 |

### 提交与 PR

每个 commit 专注于一个独立修改，message 简明扼要。提交 PR 前请确认：

- [ ] 代码编译通过（目标 SOC）
- [ ] 示例运行正确
- [ ] 测试用例通过
- [ ] 代码风格与仓库一致
- [ ] 已同步上游分支

PR 模板与 `.gitcode/PULL_REQUEST_TEMPLATE.md` 保持一致：

```markdown
# 合入来源

- [ ] 需求
- [ ] 问题单
- [ ] issue

# 问题/功能描述

# 修改方案描述

# 是否涉及UT/ST

- [ ] 是
- [ ] 否（说明理由）

# 开发自检

- [ ] 规范：未超过1k；超过1k已完成备案；反合入需要标题注明【反合】
- [ ] 规范：已关联issue或里程碑
- [ ] 规范：不需要UT/ST且已说明理由；需要且已包含UT/ST
- [ ] 安全：涉及对外接口/参数/环境变量改动，并同步修改资料
- [ ] 安全：未引入三方开源软件；引入三方开源软件并经过安全评审
- [ ] 安全：未新增对外接口/参数；新增对外接口/参数有校验
- [ ] 规范：未涉及资料改动；涉及资料改动并已完成修改
```
