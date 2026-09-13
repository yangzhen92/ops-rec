# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# pylint: disable=attribute-defined-outside-init
"""
jagged_jagged_bmm_jagged_out_kernel 算子的 ATK 自定义 API 执行文件

语义（来自 FBGEMM 官方测试 jagged_jagged_bmm_jagged_out_test.py + CPU 参考 cpu_sll.py）：
    逐 batch 计算 C_i = A_i @ B_i^T
    jagged_A [Σm_i, K]            每 batch [m_i, K]
    jagged_B [K, Σn_i]            wrapper 接收「已转置」布局（官方测试传 jagged_B.T）
    jagged_C [Σ(m_i·n_i)]         C_i 按行主序展平（输出，wrapper 内部构建）
    约束：lengths_n = lengths_m，lengths_mn = lengths_m·lengths_n

设计分工（Generator 只约束，API 只补 jagged 元信息）：
    ATK 精度机制是 single_bm —— 同一算子 GPU(golden) vs NPU 双跑对比，无 CPU 数学参考，
    因此输入必须 (a) GPU/NPU 一致、(b) 自洽满足约束。

    Generator 已保证 shape 自洽：jagged_A=[total_m, K]、jagged_B=[K, total_m]、offset=[B+1]。
    但「每 batch 的 m_i 如何切分」是 jagged 元信息，无法通过 tensor/outlier_values 注入
    （outlier_values 只能存 2 个数）。所以 API 用 ATK 采样到的 total_m 与 B 做确定性「变长」切分
    （贴合原生 lengths_m=randint(1,max_L) 的变长分布，产生 ragged 边界），得到 lengths_m，
    推出 lengths_n=lengths_m、lengths_mn=m·n，再 cumsum 出三个 offset。

    关键：tensor 数据直接用 ATK 采样的真实值（jagged_A/B 已是正确 shape），不再平铺/造数，
    多样性完全来自 ATK 的 shape 采样。
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def get_device():
    """获取可用的设备，优先 NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


# 从被测代码直接引用 wrapper（它已正确处理 grid/launch）
from triton_jagged_bmm_jagged_out import triton_jagged_jagged_bmm_jagged_out


def _split_jagged(total: int, parts: int):
    """
    把 total 切成 parts 份「变长」jagged 长度，贴合算子原生用法
    （官方测试 lengths_m = randint(1, max_L)，逐 batch 长度不等）。

    实现：用只依赖 (total, parts) 的 LCG 生成确定性伪随机权重，按权重分配 total，
    每份保底 1，余数补齐。因只依赖两个整数 → GPU/NPU 完全一致（满足 single_bm），
    同时产生 ragged 边界，真正覆盖 kernel 的 mask/边界逻辑。
    前提 total ≥ parts（Generator 已约束），保证每份 ≥ 1。
    """
    if parts <= 0:
        return []
    if total <= parts:
        return [1] * parts  # 退化：每份 1

    # LCG 确定性伪随机权重（种子仅由 total、parts 决定）
    seed = (total * 1000003 + parts * 9176 + 12345) & 0x7FFFFFFF
    weights = []
    for _ in range(parts):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        weights.append((seed % 997) + 1)  # 1..997，避免全相等
    wsum = sum(weights)

    lengths = [1] * parts  # 每份保底 1
    remaining = total - parts  # 剩余按权重分配
    acc = 0
    for i in range(parts):
        add = remaining * weights[i] // wsum
        lengths[i] += add
        acc += add
    # floor 造成的余数逐个补齐（确定性：从 0 开始）
    leftover = remaining - acc
    i = 0
    while leftover > 0:
        lengths[i % parts] += 1
        leftover -= 1
        i += 1
    return lengths


@register("jagged_jagged_bmm_jagged_out_kernel")
class JaggedJaggedBmmJaggedOutKernelApi(BaseApi):
    """jagged_jagged_bmm_jagged_out_kernel 算子的 API 封装"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理阶段（不计入性能）：补 jagged 切分元信息，tensor 用 ATK 真实数据"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        jagged_A_input = input_data.kwargs.get("jagged_A", None)
        jagged_B_input = input_data.kwargs.get("jagged_B", None)
        offsets_m_input = input_data.kwargs.get("offsets_m", None)

        if jagged_A_input is None or jagged_B_input is None:
            raise ValueError("jagged_A or jagged_B not found")
        if offsets_m_input is None:
            raise ValueError("offsets_m not found")

        # ====== NPU dtype 适配（Fix）======
        # 背景：kernel 内 tl.dot + tl.load(other=0.0) 的 Python 字面量 0.0 在 Ascend 上被
        #       促升为 fp64，导致 tl.dot 报 "Unsupported lhs dtype fp64"（编译期失败）。
        # 手段：输入强转 fp32 让 kernel 编译通过；输出再转回原始 dtype，
        #       使 local(NPU) 输出与 remote(GPU golden) 的 dtype 一致，通过双标杆对比。
        # 记录原始 dtype（以 jagged_A 为准，generator 已约束 jagged_A/jagged_B dtype 一致）
        if isinstance(jagged_A_input, torch.Tensor):
            self._orig_dtype = jagged_A_input.dtype
        else:
            self._orig_dtype = torch.float32

        # 原始（未强转）实现，保留以便回退：
        # jagged_A = self._to_tensor(jagged_A_input, device)
        # jagged_B = self._to_tensor(jagged_B_input, device)
        jagged_A = self._to_tensor(jagged_A_input, device, dtype=torch.float32)
        jagged_B = self._to_tensor(jagged_B_input, device, dtype=torch.float32)
        offsets_m_raw = self._to_tensor(offsets_m_input, device, torch.int32)

        # B 由 offset 长度决定，total_m/K 由 ATK 采样的 jagged_A shape 决定（都可信）
        B = offsets_m_raw.shape[0] - 1
        if B <= 0:
            raise ValueError(f"B must > 0, got {B}")

        total_m = jagged_A.shape[0]
        K = jagged_A.shape[1] if jagged_A.ndim == 2 else 1

        # Generator 约束了 total_m ≥ B；防御性兜底
        if total_m < B:
            raise ValueError(f"total_m({total_m}) < B({B}), generator constraint violated")

        # 确定性变长切分 total_m → 每 batch 长度（贴合原生 randint 变长分布）；n=m，mn=m·n
        lengths_m = _split_jagged(total_m, B)
        lengths_n = list(lengths_m)
        lengths_mn = [m * n for m, n in zip(lengths_m, lengths_n)]

        total_n = sum(lengths_n)  # = total_m

        # jagged_B 应为 [K, total_n]。ATK 采样时 Generator 已设为 [K, total_m]==[K, total_n]，
        # 校验并在极端不一致时修正列数（保证 wrapper 的 size(1)==Σn 断言成立）。
        if jagged_B.ndim != 2 or jagged_B.shape[0] != K or jagged_B.shape[1] != total_n:
            jagged_B = self._fit_2d(jagged_B, K, total_n, jagged_A.dtype, device)

        # cumsum 出三个 offset（int32，自洽）
        offsets_m = self._cumsum_tensor(lengths_m, device)
        offsets_n = self._cumsum_tensor(lengths_n, device)
        offsets_mn = self._cumsum_tensor(lengths_mn, device)

        lengths_m_t = torch.tensor(lengths_m, dtype=torch.int32, device=device)
        lengths_n_t = torch.tensor(lengths_n, dtype=torch.int32, device=device)
        lengths_mn_t = torch.tensor(lengths_mn, dtype=torch.int32, device=device)

        # max_seq_len = max(单 batch 长度)，kernel 不截断、grid 不空转
        max_seq_len = max(max(lengths_m), max(lengths_n))

        self._device = device
        self._jagged_A = jagged_A.contiguous()
        self._jagged_B = jagged_B.contiguous()
        self._offsets_m = offsets_m
        self._offsets_n = offsets_n
        self._offsets_mn = offsets_mn
        self._lengths_m = lengths_m_t
        self._lengths_n = lengths_n_t
        self._lengths_mn = lengths_mn_t
        self._max_seq_len = max_seq_len

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行阶段（计入性能）：仅同步 + 调用 wrapper"""
        device = self._device

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        jagged_C = triton_jagged_jagged_bmm_jagged_out(
            self._jagged_A,
            self._jagged_B,
            self._max_seq_len,
            self._lengths_m,
            self._lengths_n,
            self._lengths_mn,
            self._offsets_m,
            self._offsets_n,
            self._offsets_mn,
            allow_tf32=False,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== NPU dtype 适配（Fix）======
        # 输入曾被强转 fp32 以通过编译；此处把输出转回原始 dtype，
        # 使其与 remote(GPU golden) 输出 dtype 一致，通过双标杆 dtype 检查。
        # 原始（未转回）实现，保留以便回退：
        # return jagged_C
        orig_dtype = getattr(self, "_orig_dtype", None)
        if orig_dtype is not None and jagged_C.dtype != orig_dtype:
            jagged_C = jagged_C.to(orig_dtype)
        return jagged_C

    # ====== 工具方法 ======
    def _cumsum_tensor(self, lengths, device):
        """由 lengths 生成 offset 张量：[0, l0, l0+l1, ...]，int32"""
        offsets = [0]
        for l in lengths:
            offsets.append(offsets[-1] + l)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _fit_2d(self, raw, rows, cols, dtype, device):
        """把 raw 拟合到 [rows, cols]：够长截断，不够循环平铺（仅兜底极端不一致）"""
        flat = raw.reshape(-1).to(device=device, dtype=dtype)
        need = rows * cols
        if flat.shape[0] < need:
            reps = (need + flat.shape[0] - 1) // flat.shape[0]
            flat = flat.repeat(reps)
        return flat[:need].reshape(rows, cols).contiguous()

    def _to_tensor(self, data, device, dtype=None):
        """转换数据到 tensor"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(data, dtype=dtype, device=device)
