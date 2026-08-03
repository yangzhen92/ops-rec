# ops-rec

Ascend NPU Custom Operations for Recommendation Systems.

## 简介

ops-rec 是基于 CANN 的推荐系统自定义算子仓库，算子实现包括 AscendC 和 Triton 两种后端，打包为 pip install 即可使用的 whl 包。

- **AscendC 算子**：基于 CANN AscendC 框架实现，提供高性能自定义算子。
- **Torch 无缝集成**：通过 `TORCH_LIBRARY` 注册，通过 `torch.ops.ops_rec` 直接调用。
- **多芯片自适应**：自动探测 Atlas A2 / A3 / 950 芯片，加载对应的 OPP 产物。

```python
import ops_rec

# AscendC 算子
ops_rec.ascendc.activation.ln_mul(x, u, gamma, beta)
```

## 目录结构
```text
ops-rec
├── ascendc/                        ← AscendC 算子源码（CMake 编译单元）
│   ├── activation/                 activation类算子
│   │   └── ln_mul/                 单算子 vendor
│   ├── sequence/                   sequence 类算子(预留，revers_sequence)
│   ├── attention/                  attention 类算子（预留，hstu，fa）
│   └── common/                     共享头文件（log, utils）
├── triton/                         ← Triton算子源码（预留、无需编译）
├── experimental/                   ← 实验性算子源码（内按后端分组）
│   ├── ascendc/                    实验性 AscendC 算子（预留）
│   └── triton/                     实验性 Triton 算子
│
├── cmake/                           ← CMake 编译配置
├── scripts/                         ← 构建辅助脚本
├── docs/                            ← 设计文档和指南
├── tests/                           ← Python 集成测试
├── ops_rec/                         ← Python 包（whl 发布产物）
│
├── build.sh                         ← 统一构建入口脚本
├── setup.py                         ← scikit-build 打包配置
└── CMakeLists.txt                   ← 根 CMake 构建文件
```

## 环境部署

推荐环境：

- 操作系统：Linux
- Python >= 3.10
- CANN >= 9.0.0
- PyTorch 与 torch_npu 版本配套
- 构建依赖：`scikit-build`、`cmake`、`gcc/g++`

```bash
# CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 安装编译依赖
pip install scikit-build

# 验证 NPU 环境
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## 编译安装

```bash
# 编译所有 SOC 并安装
bash build.sh --install -j16

# 仅编译指定 SOC
bash build.sh --socs atlas_950 --install -j16
```

生成的 whl 位于 `dist/`，可直接 `pip install dist/ops_rec-*.whl`。

## 运行示例

```bash
# 运行所有 torch 示例
bash run_examples.sh --type torch

# 运行指定算子的 aclnn 示例
bash run_examples.sh --type aclnn --ops ln_mul
```

## 算子开发

两种开发模式：

| 模式 | 说明 | 示例 |
|---|---|---|
| 单算子 vendor | 一个 vendor 一个算子 | `ln_mul` |
| 多算子 vendor | 一个 vendor 多个算子，共享代码 | `hstu_v1` |

详见 [算子迁移指南](docs/contribute_guide.md)。

## License

Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
