# PLAN.md — hstu_forward_v1 修复记录

算子：hstu_forward_v1（hstu_dense / hstu_jagged / hstu_paged）
测试入口：`test/pytest/test_hstu_forward.py`

## 修复项记录

### FIX-20260817-01：get_golden 未处理 GQA（N_q != N_k）导致实机 pytest 失败

| 项目 | 内容 |
|------|------|
| issue_id | FIX-20260817-01 |
| original_problem | 实机 pytest（服务器 atlas_950）用例 `test_hstu_dense_gqa_dimmismatch`（N_q=4, N_k=2, D_qk=32, D_v=64）在 get_golden 中抛 `ValueError: operands could not be broadcast together with remapped shapes ... (1,16,4,32)->(1,4,16,newaxis,32) (1,16,2,32)->(1,2,newaxis,16,32)` |
| observed_signal | `test_hstu_forward.py::get_golden` 中 `qk = np.einsum("bnhd,bmhd->bhnm", q_f, k_f)`；q 的 head 维（N_q=4）与 k 的 head 维（N_k=2）不同，einsum 中 n/m 为不同标签不求和、无法广播，直接抛 ValueError |
| first_failed_boundary | golden 层（numpy 计算），非 kernel / 编译层 |
| classification | actual_failure |
| root_cause | get_golden 直接要求 N_q == N_k，未处理 GQA（N_k < N_q）场景 |
| 权威依据 | RecSDK-2026-Q3 `cust_op/test/hstu_dense/torch/test_hstu_jagged_forward.py` golden_op_exec（L139-140）：`h_qk_ratio = head_nums_q // head_nums_k`；`k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)`；`v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)`。即 k/v 的 head 维按 `N_q // N_k` 扩展为 N_q 个 head 后做标准 MHA（q head i 共享 k/v head i//ratio）。README 约束：N_q % N_k == 0 且 N_k == N_v |
| fix | 仅修改 `test/pytest/test_hstu_forward.py::get_golden` 一处：N_q != N_k 时先 `k_f = np.repeat(k_f, N_q // N_k, axis=2)`、`v_f = np.repeat(v_f, N_q // N_k, axis=2)` 扩展 head 维（axis=2，dim 维不扩展），并加 `N_q % N_k == 0` 防御性断言；原有 einsum（`qk = np.einsum("bnhd,bmhd->bhnm", q_f, k_f)`、`out = np.einsum("bhnm,bmhd->bnhd", qk, v_f)`）与其余逻辑保持不变；N_q == N_k 时行为不变。未改动其他用例、其他 golden 函数、接口调用与断言 |
| verification | 本地 numpy 重放（ast 提取 get_golden 独立执行，不依赖 torch/ops_rec/NPU）：① GQA 用例 q=(1,16,4,32) k=(1,16,2,32) v=(1,16,2,64) mask_type=0 silu_scale=1/16 max_seq_len=16 不再抛异常，输出形状 (1,16,4,64) ✓；② GQA 输出与显式 repeat 后的标准 MHA 公式逐元素一致（max_abs_err=0.000e+00）✓；③ 分组语义正确（不同 q head 的注意力不同）✓；④ 非 GQA（N_q==N_k，shape (2,32,4,32)）输出与修复前公式逐元素一致（max_abs_err=0.000e+00）✓；⑤ fp16 输入 GQA/非 GQA 输出形状与 dtype 保持 ✓。8 项检查全部 PASS |
| positive_regression | 上述 numpy 重放覆盖修复影响的 required case（GQA 用例）与未受影响路径（N_q==N_k），全部通过 |
| remaining_boundary | 本机为 Windows 开发环境无 NPU，未在本机重跑实机 pytest；实机用例建议在 atlas_950 服务器重新执行 `test_hstu_dense_gqa_dimmismatch` 及全量 pytest 回归 |
| skill_promotion | reusable：GQA golden 计算在 get_golden 类 numpy 实现中一律先按 `N_q // N_k` 扩展 k/v head 维（repeat_interleave 语义）再做标准 MHA，与 RecSDK golden 对齐 |

## 测试结果摘要

| 用例 | 状态 | 说明 |
|------|------|------|
| test_hstu_dense_gqa_dimmismatch | 已移除 | 实机验证（atlas_950）结论：dense 4D 路径不支持 GQA，用例与算子实际能力不符，见 FIX-20260818-01 |
| 非 GQA 用例（N_q == N_k） | numpy 重放通过 | 输出与修复前逐元素一致，行为不变 |
| README GQA 表述修正（FIX-20260818-02） | 文档核对通过 | 4 处 GQA 表述已修正为"仅 jagged 支持"：L21 特性总览、L72 k 输入行、L157-168 GQA 示例（dense 4D → hstu_jagged.equal）、L288-296 GQA 支持说明（标题+正文首句）；README 不再有"dense 支持 GQA"表述 |

## 修复项记录（追加）

### FIX-20260818-01：移除 test_hstu_dense_gqa_dimmismatch 用例（dense 4D 路径不支持 GQA）

| 项目 | 内容 |
|------|------|
| issue_id | FIX-20260818-01 |
| original_problem | 实机验证结论（服务器 atlas_950，用户反馈）：hstu_dense（dense 4D 布局）路径不支持 GQA（N_k < N_q）。测试脚本中的用例 `test_hstu_dense_gqa_dimmismatch`（GQA + qk/v dim 不等）与算子的实际能力不符 |
| classification | actual_failure（用例与算子实际能力不符，非 golden/kernel 缺陷） |
| fix | 仅修改 `test/pytest/test_hstu_forward.py` 两处：① 删除 `test_hstu_dense_gqa_dimmismatch` 用例（含其两个 `@pytest.mark.parametrize` 装饰器与函数体），不留占位；② `get_golden` 中 GQA 分支（N_k != N_q 时 np.repeat 扩展 k/v head 维）保留，仅在其注释后补充「当前 dense 4D 布局路径不支持 GQA，该分支为与 RecSDK golden 对齐的通用参考实现，供未来用例复用」。未改动其他用例、函数、接口调用与断言 |
| verification | 静态核对：文件中已无 `test_hstu_dense_gqa_dimmismatch` 定义；其余用例（test_hstu_dense_eager / test_hstu_dense_custom_mask / jagged equal/delta / paged / graph / CI 冒烟）原样保留；get_golden 的 GQA 分支代码逻辑未变（仅注释补充） |
| positive_regression | 删除的用例不再被 pytest 收集；其余用例代码未触碰，行为不变 |
| remaining_boundary | 本机为 Windows 开发环境无 NPU，未在本机重跑实机 pytest；建议在 atlas_950 服务器重新执行全量 pytest 回归确认 |
| skill_promotion | reusable：dense 4D 布局不支持 GQA 的算子，测试脚本中不应包含 N_k < N_q 的用例；GQA golden 参考实现保留在 get_golden 中供未来（如 jagged GQA）用例复用 |

### FIX-20260818-02：README GQA 支持表述修正（仅 jagged 布局支持）

| 项目 | 内容 |
|------|------|
| issue_id | FIX-20260818-02 |
| original_problem | README.md 中 GQA 相关表述未区分布局，与实机验证结论不符：L21 特性总览称"本算子支持 GQA"、L72 k 输入行写"GQA 时头数可小于 Q"、L157-168 GQA 用法示例使用 hstu_dense（dense 4D）接口、L288-296 GQA 支持说明未限定布局 |
| observed_signal | 实机验证结论（atlas_950，用户确认）：hstu_dense（dense 4D 布局）路径不支持 GQA（N_k < N_q）；hstu_jagged（jagged 布局，equal/delta）路径支持 GQA。README 表述与实机验证结论不一致 |
| first_failed_boundary | 文档层（README），非代码/编译/测试层 |
| classification | actual_failure（文档与实机行为不一致） |
| root_cause | README 按早期设计意图撰写，GQA 支持范围未按布局区分，dense 与 jagged 混为一谈 |
| fix | README.md 共 4 处修正：① L21 特性总览改为"**GQA 支持（仅 jagged）**"，注明 hstu_jagged（equal/delta）支持 GQA、hstu_dense（dense 4D 布局）不支持 GQA 且 q/k/v 头数必须一致；② L72 输入表格 k 行"同 q（GQA 时头数可小于 Q）"改为"同 q（仅 jagged 时头数可小于 Q（GQA）；dense 布局不支持 GQA，头数必须与 Q 一致）"；③ L157-168 GQA 用法示例删除 hstu_dense 4D 示例（N_q=8、N_k=2），替换为 hstu_jagged.equal jagged 版示例（3D [total, N, D] + seq_offset，seq_lens=[4,6]、total=10、N_q=8、N_k=2，9 参数 q,k,v,mask,attn_bias,mask_type,max_seq_len,silu_scale,seq_offset），约束保持 `N_q % N_k == 0` 且 `N_k == N_v`；④ L288-296 标题改为"## GQA (Grouped Query Attention) 支持说明（仅 jagged）"，正文首句注明 dense 布局不支持 GQA、q/k/v 头数必须一致。除上述 4 处外未改动 README 其他内容 |
| verification | 逐处核对通过：grep 全量检查 README 中 GQA 相关行（L21/L72/L157/L167/L294/L296/L301），不再存在"dense 支持 GQA"表述，全部限定为仅 jagged；jagged GQA 示例语法与接口核对：`hstu_jagged.equal(q, k, v, None, None, 0, S, 1.0 / S, seq_offset)` 9 参数（q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale, seq_offset）与接口签名一致；seq_lens=[4,6] → seq_offset=[0,4,10]、total=10，输出形状注释 torch.Size([10, 8, 64]) 与 (total, N_q, D) 一致 |
| positive_regression | 原有约束行（`N_q % N_k == 0` 且 `N_k == N_v`）保持不变；jagged equal/delta/paged 等其他调用示例未受影响 |
| remaining_boundary | 无（纯文档修正）。dense 路径不支持 GQA 属实机既有行为，本次无代码改动；本工程为 RecSDK 迁移工程，docs/ 下无 DESIGN.md（仅 aclnn_hstu_forward.md / torch_hstu_forward.md / PLAN.md），无对应设计文档需同步 |
| skill_promotion | operator_local：GQA 支持范围须按布局（dense/jagged/paged）分别声明，README 示例与实机验证行为保持一致 |
