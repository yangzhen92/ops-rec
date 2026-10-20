# 合入来源

- [x] 需求
- [ ] 问题单
- [ ] issue

> 需求来源：RecSDK 9 算子迁移规划《ops_rec_RecSDK_9算子迁移归属与autograd》，index_select_for_rank1_backward 归属 embedding 分组 gather_for_rank1 聚合算子（反向）。

# 问题/功能描述

将 RecSDK 的 `index_select_for_rank1_backward` 算子迁移至 ops-rec。该算子是 rank-1 嵌入查表前向 `gather_for_rank1`（`y = x[index]`）的反向传播，语义等价于：

```python
grad_x = torch.zeros_like(x)
grad_x.scatter_add_(0, index, grad_y)
```

用于 embedding 权重表梯度累加，迁移前 ops-rec 缺少该能力。

# 修改方案描述

- 新建 `ascendc/embedding/` 分组与 `gather_for_rank1` 聚合 vendor（本 MR 先迁移反向；前向迁移完成后纳入同一聚合目录）
- op_host：移植 RecSDK 原版 tiling（totalLen/xDim0/baseLen/tailSplitIndex/stride）与 InferShape（grad_x 与 x 同形）；OpDef 支持 ascend910b / ascend910_93 / ascend950（950 由 SUPPORT_950 宏保护）
- op_kernel：保留原版算法——index/grad_y 分批搬入 UB，UB 内以全量 grad_x 缓冲（清零后 scatter-add），最后整块 AtomicAdd 写回 GM 实现多核累加
- torch_extension：注册 ops_rec 命名空间 schema + PrivateUse1（NPU）+ Meta（torch.compile shape 推断）；接口返回 `(grad_x, grad_index)`，grad_index 恒为 undefined，保持 RecSDK 解包语义；内部以 zeros_like 初始化 grad_x 满足 AtomicAdd 累加前提
- 注册 atlas_950 / atlas_a3 / atlas_a2 三个 SOC 白名单（cmake/soc_ops_config.cmake）
- 配套交付：aclnn + torch 双调用示例、pytest 测试（shape×dtype 矩阵、torch.compile 图模式、异常入参、CI 冒烟）、README 与双接口文档，并更新 docs/zh/ops_rec_api.md 算子索引

变更规模：19 个文件，+1344/-1（功能代码约 690 行、测试 198 行、示例 228 行、文档 232 行）。

# 是否涉及UT/ST

- [x] 是
- [ ] 否（说明理由）

# 开发自检

- [x] 规范：未超过1k；超过1k已完成备案；反合入需要标题注明【反合】（总变更 +1344/-1 超过 1k，已完成备案；其中功能代码约 690 行）
- [x] 规范：已关联issue或里程碑
- [x] 规范：不需要UT/ST且已说明理由；需要且已包含UT/ST（pytest 用例位于算子目录 test/pytest/）
- [x] 规范：pytest测试用例需包含@pytest.mark.ci标记，以便CI流水线识别并执行；set_device默认值必须为0（CI 冒烟用例 `test_index_select_for_rank1_backward_ci` 带 @pytest.mark.ci；`DEVICE = "npu:0"` 且模块级 `torch.npu.set_device(DEVICE)`）
- [x] 安全：涉及对外接口/参数/环境变量改动，并同步修改资料（新增 torch/aclnn 对外接口，已同步更新 README、docs/、docs/zh/ops_rec_api.md）
- [x] 安全：未引入三方开源软件；引入三方开源软件并经过安全评审
- [x] 安全：未新增对外接口/参数；新增对外接口/参数有校验（torch 扩展层：非空/同设备/1D/grad_y 与 index 等长校验；aclnn：dtype/format 校验）
- [x] 规范：未涉及资料改动；涉及资料改动并已完成修改
