# 构建参数说明

## 简介

`build.sh` 是 ops-rec 的统一构建入口脚本，默认在项目根目录下。脚本将 AscendC 算子源码通过 CANN 编译框架与 scikit-build 编译、打包，最终生成可 `pip install` 的 whl 包，并支持多 SOC 变体合并打包。

构建流程核心链路：

```text
build_single_soc()                      ← 对每个 SOC 变体依次执行
  └─ pip wheel . --no-build-isolation
       └─ skbuild
            ├─ cmake configure (CANN fwk_modules + 算子 CMakeLists)
            ├─ make (kernel .o/.json, op_api .so, torch 适配 .so)
            ├─ cmake --install (安装到 ops_rec/opp/<variant>/)
            └─ 生成单 SOC whl

merge_wheel.py --build-final            ← 合并所有 SOC 产物为多 SOC whl
  └─ dist/ops_rec-<version>-<py>-<abi>-<plat>.whl
```

最终 whl 安装后，`import ops_rec` 会自动按本机 SOC 选择对应 variant 加载。

## 使用方法

### 1. 配置环境变量

```bash
# 默认路径安装，以 root 用户为例
source /usr/local/Ascend/cann/set_env.sh
```

### 2. 构建命令格式

```bash
bash build.sh [--install] [-j<N>] [--socs <list>]
```

全量参数含义参见下方 [参数说明](#参数说明) 章节，请按实际情况选择合适的参数。

### 3. 常用场景

```bash
# 完整构建并安装（编译三个 SOC 变体并合并为多 SOC whl）
bash build.sh --install -j16

# 仅编译指定 SOC，不安装
bash build.sh --socs atlas_950 -j16

# 编译多个指定 SOC 并安装
bash build.sh --socs atlas_a2,atlas_a3 --install -j16

# 运行算子示例（需先安装 ops_rec whl）
bash run_examples.sh [-j<N>] [--type aclnn|torch|all] [--ops <list>]
```

## 参数说明

build.sh 共支持以下 4 个 CLI 选项，均为可选：

| 参数名 | 缩写 | 是否必选 | 默认值 | 说明 |
|---|---|---|---|---|
| `--install` | — | 否 | 不安装（仅产出 whl） | 编译完成后自动 `pip install` 生成的 whl，安装前会自动 `pip uninstall -y ops_rec` |
| `-j<N>` / `-j <N>` | — | 否 | `CMAKE_BUILD_PARALLEL_LEVEL`（缺省 CPU 核数 / 2） | 指定编译线程数；支持 `-j16`（连写）或 `-j 16`（空格分隔）两种形式 |
| `--socs <list>` / `--socs=<list>` | — | 否 | `atlas_950,atlas_a2,atlas_a3`（全量编译） | 指定 SOC 变体列表，英文逗号分隔（如 `--socs atlas_950` 或 `--socs atlas_a2,atlas_a3`） |
| `--help` / `-h` | — | 否 | — | 打印脚本使用帮助信息并退出 |

> 上述四个选项可任意组合，顺序不限；未识别的参数会被忽略并打印 warning。

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN 安装路径 | `/usr/local/Ascend/latest` |
| `OPS_REC_FORCE_VARIANT` | 强制指定运行时 SOC variant | 自动检测 |
| `CMAKE_BUILD_PARALLEL_LEVEL` | 编译并行度 | CPU 核数 / 2 |
| `CMAKE_GENERATOR` | CMake 生成器 | `Unix Makefiles`|

## 编译产物结构

> 本节展示 whl 安装后 Python 包 `ops_rec/` 的内部布局，区别于源码仓库根目录。

whl 安装后 `ops_rec/` 目录布局：

```text
ops_rec/
├── __init__.py                     # import 时自动 SOC 检测 + 加载 .so
├── env_setup.sh                    # 可选 shell 级 OPP 路径预配置
├── ascendc/                        # AscendC 算子的 Python 代理
├── ops_rec_atlas_950.so            # torch 适配层（每个 variant 一份）
├── ops_rec_atlas_a2.so
├── ops_rec_atlas_a3.so
└── opp/
    ├── atlas_950/
    │   └── vendors/
    │       ├── config.ini          # load_priority=ln_mul
    │       └── ln_mul/
    │           ├── op_api/         # libcust_opapi.so + aclnn_*.h
    │           ├── op_impl/ai_core/tbe/  # kernel .o/.json + op_tiling
    │           └── config/         # aic-*-ops-info.json
    ├── atlas_a2/...
    └── atlas_a3/...
```

## SOC 编译配置

算子编译列表按 SOC 芯片管理，文件位于 `cmake/soc_ops_config.cmake`：

```cmake
set(OPS_REC_OPS_ATLAS_950  ln_mul)
set(OPS_REC_OPS_ATLAS_A3   ln_mul)
set(OPS_REC_OPS_ATLAS_A2   ln_mul)
```

新增算子时，仅需在对应 SOC 的列表追加 vendor 名，根 `CMakeLists.txt` 无需改动。

## 运行时 SOC 检测

`ops_rec/__init__.py` 在 `import ops_rec` 时按以下顺序确定 variant：

1. 检查 `OPS_REC_FORCE_VARIANT` 环境变量（优先级最高，常用于 CI / 测试）。
2. 调用 `acl.get_soc_name()` 检测芯片型号，按 `Ascend950*` → `atlas_950`、`Ascend910B*` → `atlas_a2`、`Ascend910_93*` → `atlas_a3` 映射。
3. 将 `ASCEND_CUSTOM_OPP_PATH` 指向 `opp/<variant>/vendors/`。
4. 通过 `torch.ops.load_library` 加载对应 variant 的 `ops_rec_atlas_<variant>.so`。

> 若需要在 shell 内为其他进程预先配置 OPP 路径（绕过 SOC 检测），可执行：
> `source $(python3 -c "import ops_rec; print(ops_rec.env_setup_path())")`

## 常见问题

### Kernel 产物未出现在 whl 中

每个 kernel target 在根 `CMakeLists.txt` 中通过 `install(CODE ...)` 将产物拷贝到 `opp/<variant>/vendors/<vendor>/op_impl/ai_core/tbe/`。如未出现：

1. 确认 vendor 名已加入 `cmake/soc_ops_config.cmake` 中对应 SOC 的列表。
2. 检查 `_skbuild/<soc>/install_manifest.txt` 是否包含 `opp/<variant>/vendors/<vendor>/op_impl/ai_core/tbe/kernel/<soc>/` 下的 `.o` 与 `config/` 下的 `.json`。
3. 若 `install_manifest.txt` 中没有 kernel 路径，说明该 SOC 没有命中 vendor 列表或 kernel target 未注册。

### 编译时找不到 ops_log.h

在算子的 `op_host/CMakeLists.txt` 中确保通过 `${MY_COMMON_DIR}` 包含 common 路径：

```cmake
target_include_directories(cust_opapi PRIVATE
    ${MY_COMMON_DIR}/log
    ${MY_COMMON_DIR}/utils
)
```

`MY_COMMON_DIR` 通常为 `${CMAKE_CURRENT_SOURCE_DIR}/../../../common`。

### `torch.ops.ops_rec.ln_mul` 报 AttributeError

按以下顺序排查：

1. `import ops_rec` 时控制台是否有 `Failed to load ops_rec_atlas_<variant>.so` 警告？有则说明 `.so` 缺失或链接失败，需重新 `bash build.sh --install`。
2. 检查 `OPS_REC_FORCE_VARIANT` 是否设置了与本机不符的 variant。
3. 确认 whl 中确实打包了 `ops_rec/ops_rec_atlas_<variant>.so`：
   `python3 -c "import ops_rec, os; print(os.path.dirname(ops_rec.__file__))"` 查看目录下是否有对应 `.so`。
4. 确认算子是否在 `cmake/soc_ops_config.cmake` 中对应 SOC 的编译列表内（未编译则 `TORCH_LIBRARY` 中没有该算子的 schema）。

### `site-packages` 残留旧版本

`build.sh --install` 在安装新 whl 前会自动 `pip uninstall -y ops_rec`。如手动安装，建议：

```bash
pip uninstall -y ops_rec
pip install dist/ops_rec-*.whl --force-reinstall --no-deps
```
