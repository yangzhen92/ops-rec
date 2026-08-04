# 构建说明

## 简介

`build.sh` 是本项目的统一构建入口脚本，位于项目根目录。其功能是将 AscendC 算子源码通过 CANN 编译框架和 scikit-build 编译、打包为 pip install 即可使用的 whl 包，并支持多 SOC 变体的合并打包。

构建流程核心链路：

```text
build_single_soc()  ← 对每个 SOC 变体依次执行
  └─ pip wheel . --no-build-isolation
       └─ skbuild
            ├─ cmake configure (CANN fwk_modules + 算子 CMakeLists)
            ├─ make (kernel .o/.json, op_api .so, torch 适配 .so)
            ├─ cmake --install (安装到 ops_rec/opp/)
            └─ 生成单 SOC whl

merge_wheel.py --build-final  ← 合并所有 SOC 产物为多 SOC whl
  └─ dist/ops_rec-<version>.whl
```

## 环境要求

| 组件 | 要求 |
|---|---|
| Python | >= 3.10 |
| CANN | >= 9.0.0，需配置 `ASCEND_HOME_PATH` 环境变量 |
| torch | 已安装且可导入 |
| torch_npu | 与 torch 版本配套 |
| scikit-build | `pip install scikit-build` |
| CMake | 由 scikit-build 自动管理 |
| 编译器 | gcc / g++ 支持 C++17 |

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN 安装路径 | `/usr/local/Ascend/latest` |
| `OPS_REC_FORCE_VARIANT` | 强制指定运行时 SOC variant（测试用） | 自动检测 |
| `CMAKE_BUILD_PARALLEL_LEVEL` | 编译并行度 | CPU 核数 / 2 |

## 命令参考

### 完整参数

```bash
bash build.sh [--install] [-j<N>] [--socs <list>]
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--install` | 编译完成后自动 pip install 生成的 whl |
| `-j<N>` | 编译线程数，如 `-j16` |
| `--socs` | SOC 变体列表，逗号分隔。默认 `atlas_950,atlas_a2,atlas_a3` |

### 常用场景

```bash
# 完整构建并安装（编译三个 SOC 变体，合并为多 SOC whl）
bash build.sh --install -j16

# 仅编译指定 SOC，不安装
bash build.sh --socs atlas_950 -j16

# 运行算子示例（需先安装 ops_rec whl）
bash run_examples.sh [-j<N>] [--type aclnn|torch|all] [--ops <list>]
```

## SOC 编译配置

算子编译列表按 SOC 芯片管理，文件位于 `cmake/soc_ops_config.cmake`：

```cmake
set(OPS_REC_OPS_ATLAS_950  ln_mul hstu_v1)
set(OPS_REC_OPS_ATLAS_A3   ln_mul)
set(OPS_REC_OPS_ATLAS_A2   ln_mul)
```

新增算子时，在此文件中添加到对应 SOC 的列表即可，无需修改根 CMakeLists.txt。

## 运行时 SOC 检测

`ops_rec/__init__.py` 在 `import ops_rec` 时自动执行 SOC 检测：

1. 检查 `OPS_REC_FORCE_VARIANT` 环境变量（优先级最高）
2. 调用 `acl.get_soc_name()` 检测芯片型号
3. 根据映射表确定 variant 名
4. 设置 `ASCEND_CUSTOM_OPP_PATH` 指向 `opp/<variant>/vendors/`
5. 通过 `torch.ops.load_library` 加载对应 variant 的 `.so`

## 常见问题

### CMAKE_GENERATOR 必须设为 Unix Makefiles

CANN 的 `npu_op_kernel_library` 在生成的 Makefile 中使用 `$(MAKE)` 语法，与 Ninja 不兼容。`build.sh` 中已默认设置 `CMAKE_GENERATOR=Unix Makefiles`。

### Kernel 产物未出现在 whl 中

检查 `_skbuild/` 目录下 `install_manifest.txt` 是否包含 kernel 文件（`.o`、`.json`）。如果没有，确认算子是否在 `soc_ops_config.cmake` 的编译列表中。

### 编译时找不到 ops_log.h

在算子的 `op_host/CMakeLists.txt` 中确保包含 common 路径：

```cmake
target_include_directories(cust_opapi PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
)
```

### torch.ops.ops_rec.ln_mul 报 AttributeError

确认 `import ops_rec` 时无警告信息。如果 `.so` 不在 whl 中，检查 `cmake_install_dir` 是否与 `install(TARGETS DESTINATION .)` 正确配合。

### site-packages 有旧文件残留

`build.sh --install` 在安装新 whl 前会自动卸载旧版本。如果手动安装，建议先卸载：

```bash
pip uninstall -y ops_rec
pip install dist/ops_rec-*.whl --force-reinstall --no-deps
```

### 实验性 Triton 算子未出现在 whl 中

`experimental/triton/` 下的源码在构建时自动同步到 `ops_rec/experimental/triton/`（仅同步 `.py` 文件，不含 `test/`），由 `build_single_soc()` 函数自动完成。构建完成后这些临时文件会被自动删除。
