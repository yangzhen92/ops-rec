<h1 align="center">ops-rec</h1>

<div align="center">

[![License](https://badgen.net/badge/License/Apache-2.0/blue)](LICENSE)

</div>

## ℹ️ 简介

ops‑rec 是面向大规模推荐场景的高性能算子库，基于昇腾 CANN 平台深度定制，采用 AscendC 原生静态算子 + Triton JIT 动态可编程算子的双后端架构。算子库聚焦推荐模型核心算力瓶颈，覆盖特征预处理、多特征融合、矩阵计算、高阶非线性计算等高频场景，为推荐业务提供全链路硬件级加速支撑能力。

- **双引擎异构算子架构**：统一上层调用接口，同时支持 AscendC 极致静态高性能算子与 Triton JIT 动态灵活算子开发部署，兼顾极致性能与业务快速迭代能力，适配不同开发与上线诉求。
- **推荐专属深度优化**：针对推荐场景算子融合、HSTU 高阶复杂计算等核心性能瓶颈，完成昇腾硬件深度适配与定制重构。通过计算粒度拆解、访存合并、流水级并行优化，突破原生通用算子的性能上限，显著降低推荐模型延时、提升整机吞吐。
- **极简轻量接入**：屏蔽昇腾底层硬件、编译及平台适配细节，提供统一易用的 Python 接口，开箱即用，业务可低成本完成迁移，快速落地硬件加速收益。

## ⚙️ 功能介绍

| 功能 | 描述 | 文档 |
| --- | --- | --- |
| AscendC 静态高性能算子 | 面向推荐核心场景的高性能静态算子及其 aclnn 接口，覆盖特征预处理、多特征融合、矩阵计算、高阶非线性计算等 | [构建指南](docs/build_guide.md) |
| Triton JIT 动态算子 | 面向推荐核心场景的 Triton 动态可编程算子，支持 JIT 编译与快速迭代开发部署 | [算子贡献指南](docs/contribute_guide.md) |
| 统一 Python 调用接口 | 屏蔽 SOC 差异与编译细节，`import ops_rec` 自动按本机 SOC 加载对应算子实现 | [运行时 SOC 检测](docs/build_guide.md#运行时-soc-检测) |
| 多 SOC 构建与打包 | 一套源码编译全部 SOC 变体并合并为多 SOC whl，安装后自动选择当前芯片对应实现 | [编译产物结构](docs/build_guide.md#编译产物结构) |

## 🚀 快速入门

### 环境要求

| 组件 | 要求 |
| --- | --- |
| 操作系统 | Linux |
| Python | >= 3.10 |
| CANN | >= 9.0.0，需执行 `set_env.sh` 配置环境变量 |
| PyTorch + torch_npu | 2.7.1 / 2.10.0|
| triton + triton-ascend | 与 torch_npu 版本配套：torch_npu 2.7.1 建议配套 3.2.2，torch_npu 2.10.0 建议配套 3.6.0 |
| 构建依赖 | gcc/g++（支持 C++17）、cmake、python3（含 torch/torch_npu）；打包依赖 `packaging` |

部署前请确保驱动和固件安装正确，且 `npu-smi info` 能识别芯片及设备状态。CANN 和 TorchNPU 的安装方式请参考：

- [CANN 下载与安装](https://www.hiascend.com/cann/download)
- [Ascend for PyTorch 安装说明](https://gitcode.com/Ascend/pytorch/blob/master/README.md)

### 安装 ops_rec

算子依赖 CATLASS 源码，编译前需要初始化 submodule：

```bash
git submodule update --init --recursive
```

```bash
# 默认编译三个 SOC 变体并安装（推荐）：atlas_950、atlas_a2、atlas_a3
bash build.sh --install

# 仅编译指定 SOC
bash build.sh --socs atlas_950 --install

# 编译多个指定 SOC 并安装
bash build.sh --socs atlas_a2,atlas_a3 --install
```

生成的 whl 位于 `dist/`，可直接 `pip install dist/ops_rec-*.whl`。多 SOC 产物合并打包说明见 [构建指南](docs/build_guide.md)。

### 运行示例

```bash
# 运行所有 torch 示例
bash run_examples.sh --type torch

# 运行指定算子的 aclnn + torch 示例
bash run_examples.sh --ops ln_mul
```

`run_examples.sh` 支持 `--type aclnn|torch|all`、`--ops <list>`，要求 ops_rec 已安装（`bash build.sh --install`）。

## 📖 使用指南

### 构建变体与硬件对应关系

`build.sh` 通过 `--socs <list>` 指定构建变体，英文逗号分隔。默认编译 `atlas_950`、`atlas_a2`、`atlas_a3` 三个变体（等价于 `--socs atlas_950,atlas_a2,atlas_a3`），各变体与芯片型号的对应关系如下：

| 构建变体 | 芯片型号 | 默认编译 |
| --- | --- | --- |
| `atlas_950` | Ascend950* | ✅ |
| `atlas_a2` | Ascend910B* | ✅ |
| `atlas_a3` | Ascend910_93* | ✅ |

> 运行时 `import ops_rec` 会通过 `acl.get_soc_name()` 自动检测芯片型号并按上述映射加载对应算子实现；也可通过环境变量 `OPS_REC_FORCE_VARIANT` 强制指定运行时变体。

完整参数说明见 [构建指南](docs/build_guide.md)。

### 源码结构

```text
ops-rec/                                  # 项目根目录
├── src/                                  # 算子源码（含 ascendc/ 与 triton/ 两个后端目录）
├── cmake/                                # CMake 模块（soc_config、toolchain 等）
├── docs/                                 # 开发指南和参考文档
├── examples/                             # 算子样例
├── experimental/                         # 实验性非正式算子
├── ops_rec/                              # pip 包主体
├── pre-commit/                           # pre-commit 配置
├── scripts/                              # 构建辅助脚本（merge_wheel.py 等）
├── tests/                                # UT 测试架构文件
├── third_party/                          # 三方软件
├── build.sh                              # 统一构建入口
├── run_examples.sh                       # 示例运行入口
└── CMakeLists.txt                        # 顶层 CMake 入口
```

> 本节展示的是**源码仓库根目录**结构。whl 安装后 Python 包 `ops_rec/` 的内部布局见 [构建指南](docs/build_guide.md#编译产物结构)。

## 🛠️ 算子开发

新增算子时，按 `<group>/<vendor>` 目录约定放置代码即可。完整迁移流程、SOC 注册、`torch_extension` 写法、PR 模板等见 [算子贡献指南](docs/contribute_guide.md)。

## 🛠️ 贡献指南

欢迎参与项目贡献，贡献流程和规范请参见 [算子贡献指南](docs/contribute_guide.md)。

## ⚖️ 相关说明

🔹 [构建指南](docs/build_guide.md)<br>
🔹 [算子列表](docs/zh/ops_rec_api.md)<br>
🔹 [算子贡献指南](docs/contribute_guide.md)<br>
🔹 [Apache License 2.0 许可证](LICENSE)<br>

## 🤝 建议与交流

欢迎大家通过以下方式提出问题、交流讨论。

| 资源 | 说明 |
| --- | --- |
| [创建Issue](https://gitcode.com/Ascend/ops-rec/issues/new) | 提交 Bug、需求或建议 |

## 🙏 致谢

感谢所有为 ops-rec 和昇腾开源社区贡献代码、文档、测试、问题反馈与技术建议的开发者。欢迎更多社区伙伴参与共建！
