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
| test_hstu_dense_gqa_dimmismatch | numpy 重放通过 | golden 不再抛异常，输出形状 (1,16,4,64)，语义与 RecSDK 一致 |
| 非 GQA 用例（N_q == N_k） | numpy 重放通过 | 输出与修复前逐元素一致，行为不变 |
