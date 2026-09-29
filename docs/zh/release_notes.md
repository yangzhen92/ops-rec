# 版本说明书

ops-rec 26.2.0 是本项目首次发布的 Release 版本，核心要点如下：

## 关键特性

- 面向大规模推荐场景的高性能算子库，采用 AscendC 原生静态算子 + Triton JIT 动态算子的双后端架构，本版本交付 AscendC 算子。
- 提供 activation、attention、mixing、sequence 等算子，覆盖特征预处理与融合、HSTU 高阶注意力、块稀疏与掩码等推荐核心场景。
- 支持 Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品，运行时按 SOC 自动加载对应算子实现。
- 提供统一 Python 调用接口（`ops_rec.ascendc.<group>.<op_name>`）与 C 语言 aclnn 接口。
- 配套 CANN 9.2.0、Ascend HDK 26.2.0、TorchNPU 26.2.0、PyTorch 2.10.0+cpu。

## 版本配套说明

### 产品版本信息

| 产品名称 | 产品版本 | 版本类型 |
| --- | --- | --- |
| ops-rec | 26.2.0 | Release版本 |

### 相关产品版本配套说明

**表 1** ops-rec软件版本配套表

| ops-rec | CANN版本 | Ascend HDK版本 | TorchNPU版本 | PyTorch版本 |
| --- | --- | --- | --- | --- |
| 26.2.0 | 9.2.0 | 26.2.0 | 26.2.0 | 2.10.0+cpu |

> [!NOTE]
>
> 不同硬件支持的算子范围存在差异，部分 HSTU V2、块稀疏及任意函数掩码算子仅支持 Ascend 950PR系列产品，具体支持情况请参见[算子列表](ops_rec_api.md)。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中"/"表示不可配套，"Y"表示可配套。
>
> 本版本为首次发布，不涉及历史版本间的兼容性或升级路径。

**表 2** ops-rec 26.2.0版本兼容性

| ops-rec | CANN 9.2.0 | Ascend HDK 26.2.0 | TorchNPU 26.2.0 | PyTorch 2.10.0+cpu |
| --- | --- | --- | --- | --- |
| 26.2.0 | Y | Y | Y | Y |

## 版本使用注意事项

- 使用前需完成对应版本的昇腾驱动、固件、CANN 及 PyTorch、TorchNPU 安装。
- `import ops_rec` 会通过 `acl.get_soc_name()` 自动检测芯片型号并加载对应算子实现，也可通过环境变量 `OPS_REC_FORCE_VARIANT` 强制指定运行时变体。
- 各 SOC 编译哪些算子由 `cmake/soc_ops_config.cmake` 白名单控制，构建时未列入白名单的算子在对应芯片上不可用。
- 默认构建 `atlas_950`、`atlas_a2`、`atlas_a3` 三个变体，也可通过 `build.sh --socs` 指定目标变体。
- 算子名称和接口与上游保持一致，但部分算子的数据类型、输入形状存在硬件相关约束，使用前请查阅对应算子文档。

## 更新说明

### 新增特性

| 特性名称 | 特性描述 | 配套产品 |
| --- | --- | --- |
| activation 激活算子 | ln_mul（LayerNorm + Multiply 融合，归一化与特征缩放融合）、in_linear_silu（Linear + Silu 融合及反向，HSTU 注意力前处理）。 | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品 |
| attention 注意力算子 | HSTU V1 前向与反向，覆盖 dense、jagged、paged 三种布局，支持 GQA 与 dim 不等；HSTU V2 反向（统一 jagged 格式，支持窗口掩码）；DeBERTa 解耦注意力，支持 c2p/p2c 与双向位置编码。 | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品（HSTU V2 仅 Ascend 950PR系列产品） |
| 块稀疏与掩码算子 | 块稀疏张量生成（Q2K/K2Q 正反向、块索引无同步压缩）、任意函数掩码 af 编码（HSTU/SLA）、Flash Attention 任意区间掩码推理。 | Ascend 950PR系列产品 |
| mixing 混合算子 | token_mixing（Token 混合归一化）。 | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品 |
| sequence 序列算子 | reverse_sequence（按 seq_lengths 对每个 batch 有效行逆序）。 | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品 |
| 统一接口与多 SOC 构建 | 提供统一 Python 调用接口与 C 语言 aclnn 接口；支持一套源码构建 `atlas_950`、`atlas_a2`、`atlas_a3` 多 SOC 变体并合并打包，安装后按 SOC 自动选择对应实现。 | Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR系列产品 |

### 业务接口变更

本版本为首次发布，不涉及业务接口变更。

### 关键特性变更

本版本为首次发布，不涉及关键特性变更。

### 已解决的问题

无。

### 遗留问题

- Triton 算子与实验性算子将在后续版本补充。

## 升级影响

### 升级过程对现行系统的影响

本版本为首次发布，不涉及从历史版本升级。

### 升级后对现行系统的影响

本版本为首次发布，不涉及从历史版本升级。

## 26.2.0版本配套文档

| 文档名称 | 内容简介 | 更新说明 |
| --- | --- | --- |
| [ops-rec README](../../README.md) | 介绍项目能力、环境要求、构建安装与基本使用方式。 | 26.2.0首次发布。 |
| [构建指南](../build_guide.md) | 介绍多 SOC 构建、打包与运行时 SOC 检测机制。 | 26.2.0首次发布。 |
| [算子列表](ops_rec_api.md) | 介绍算子分组、Python 调用接口及硬件支持情况。 | 26.2.0首次发布。 |
| [算子贡献指南](../contribute_guide.md) | 介绍新增算子的目录约定、SOC 注册与开发流程。 | 26.2.0首次发布。 |

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表

无。
