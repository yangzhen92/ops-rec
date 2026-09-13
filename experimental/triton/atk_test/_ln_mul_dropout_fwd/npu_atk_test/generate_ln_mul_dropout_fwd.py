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

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("generate_ln_mul_dropout_fwd")
class LnMulDropoutFwdGenerator(CaseGenerator):
    """_ln_mul_dropout_fwd 算子参数约束生成器"""

    # Wrapper BLOCK_D 上限（BLOCK_D = min(65536//elem_size, next_pow2(D))）
    # 参考：/test/h00930401/skill/ATKtest/guides/系统化分析.md 30%小/70%大设计。
    # _MAX_D 仅做硬上界钳制（真实上界由 YAML dim_values 决定，已压到 1024）；
    # _MIN_D 提到 64，避免 BLOCK_D=16 的"搬运型" case（算力吃不饱、耗时 <10µs）。
    _MAX_D_FP32 = 65536 // 4  # 16384
    _MAX_D_FP16 = 65536 // 2  # 32768
    _MIN_D = 64  # 每行至少 64 列，保证单 kernel 有足够并行度避开 <10µs
    _MAX_U_HEADS = 16  # 3D 分支时 H 上限，控制规模

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if not hasattr(case_config, "inputs") or len(case_config.inputs) < 6:
            return case_config

        x = case_config.inputs[0]
        u = case_config.inputs[1]
        weight = case_config.inputs[2]
        bias = case_config.inputs[3]
        # inputs[4]=eps, inputs[5]=dropout_ratio（scalar，不涉及 shape 约束）
        # inputs[6]=training, inputs[7]=concat_ux, inputs[8]=seed（attr/scalar）

        # ========== 解析基础维度 ==========
        N = self._resolve_N(x)
        D = self._resolve_D(x)

        # ========== 强制覆盖 x / weight / bias 的 shape ==========
        self._set_shape(x, [N, D])
        self._set_shape(weight, [D])
        self._set_shape(bias, [D])

        # ========== u：2D 或 3D 分支 ==========
        # 读 ATK 候选：shape 可能是 [_,_] 或 [_,_,_]（ATK 的 list）
        # 关键约束（wrapper 内核三角决定）：
        #   3D 分支（NUM_U_SLICES>1）使用 tl.arange(0, NUM_U_SLICES) 与
        #   tl.arange(0, UD)，二者都必须是非零 2 的幂，且 NUM_U_SLICES*UD==D；
        #   同时 D==BLOCK_D（BLOCK_D=next_power_of_2(D)）才能与 y 逐元素乘。
        #   → 3D 分支只在 【D 是 2 的幂】 时启用；否则一律降级为 2D [N, D]。
        u_ndim = self._get_ndim(u)
        if u_ndim == 3 and self._is_power_of_two(D):
            # 3D：u=[N, H, UD]，要求 H/UD 均为 2 的幂且 H*UD==D
            H, UD = self._resolve_u_3d(u, D)
            if H * UD == D and self._is_power_of_two(H) and self._is_power_of_two(UD):
                self._set_shape(u, [N, H, UD])
            else:
                # 解析结果不满足（如 H/UD 非 2 的幂），确定性降级 2D
                self._set_shape(u, [N, D])
        else:
            # 2D：u=[N, D]（NUM_U_SLICES=1，UD=D）
            self._set_shape(u, [N, D])

        # ========== dtype 强制对齐 x ==========
        # u/weight/bias 与 x 逐元素乘/加权，语义要求同 dtype；若不一致按 x 对齐
        if hasattr(u, "dtype") and hasattr(x, "dtype"):
            u.dtype = x.dtype
        if hasattr(weight, "dtype") and hasattr(x, "dtype"):
            weight.dtype = x.dtype
        if hasattr(bias, "dtype") and hasattr(x, "dtype"):
            bias.dtype = x.dtype

        return case_config

    # ========== 维度解析（确定性） ==========

    def _resolve_N(self, x):
        """从 x.shape[0] 解析 N，或确定性默认 64"""
        v = self._get_dim(x, 0)
        if v is not None and v > 0:
            return v
        return 64

    def _resolve_D(self, x):
        """从 x.shape[1] 解析 D；钳制在 [MIN_D, MAX_D]，防止 next_pow2 溢出"""
        v = self._get_dim(x, 1)
        if v is None or v <= 0:
            return 512

        dtype = getattr(x, "dtype", None)
        max_d = self._MAX_D_FP16 if dtype in ("fp16", "bf16") else self._MAX_D_FP32
        v = max(self._MIN_D, min(v, max_d))
        return v

    def _resolve_u_3d(self, u, D):
        """3D 分支：从候选解析 H、UD。

        要求（wrapper 内核 tl.arange 三角）：
          - H（=NUM_U_SLICES）与 UD 均为非零 2 的幂
          - H * UD == D
        D 必须为 2 的幂（调用处已保证）。返回 (H, UD) 保证两条件。
        """
        h = self._get_dim(u, 1)
        ud = self._get_dim(u, 2)

        if h is not None and ud is not None and h > 0 and ud > 0:
            if self._is_power_of_two(h) and self._is_power_of_two(ud) and h * ud == D:
                return h, ud

        # 确定性回退：H 取 D 的最大 <= MAX_U_HEADS 的 2 的幂因子（>=1）；
        # D 为 2 的幂时 UD=D//H 自动也为 2 的幂，故 H*UD==D 且均 2 的幂。
        H = self._largest_pow2_factor_le(D, self._MAX_U_HEADS)
        return H, D // H

    def _largest_pow2_factor_le(self, n, limit):
        """n 的最大 <= limit 的非零 2 的幂因子（最低返回 1）"""
        best = 1
        p = 1
        while p <= min(n, limit):
            if n % p == 0:
                best = p
            p *= 2
        return best

    @staticmethod
    def _is_power_of_two(v):
        """非零整数且为 2 的幂"""
        return v > 0 and (v & (v - 1)) == 0

    # ========== 通用安全取值 ==========

    def _get_ndim(self, tensor):
        shape = getattr(tensor, 'shape', None)
        if isinstance(shape, list):
            return len(shape)
        return None

    def _get_dim(self, tensor, idx):
        shape = getattr(tensor, 'shape', None)
        if isinstance(shape, list) and len(shape) > idx:
            v = shape[idx]
            return v if isinstance(v, int) else None
        return None

    def _set_shape(self, tensor, target_shape):
        """安全覆盖 shape（ATK 的 shape 为 list）"""
        if hasattr(tensor, 'shape') and isinstance(tensor.shape, list):
            tensor.shape[:] = target_shape
        else:
            tensor.shape = target_shape
