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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 原生 wrapper（API 直接调用它，不复制其内部逻辑）
from triton_norm_mul_dropout import triton_layer_norm_mul_dropout_fwd

# 元素字节数（用于 D 上限校验）
_ELEMENT_SIZE = {'fp32': 4, 'fp16': 2, 'bf16': 2}


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


def _derive_seed(block_d: int) -> int:
    """
    从 BLOCK_D 确定性派生固定 seed（0 <= seed < 2^62）。

    说明：ATK 注入的 seed 是带随机性的，直接传给 wrapper 会让 GPU/NPU 的
    dropout 掩码基于不同的随机种子，导致两边输出不一致。这里用确定性的
    Knuth 乘法哈希将 shape 信息映射为一个固定 seed，两个平台完全一致。
    """
    raw = block_d * 2654435761 + 12345
    return raw % (2 ** 62)


@register("_ln_mul_dropout_fwd")
class LnMulDropoutFwdApi(BaseApi):
    """_ln_mul_dropout_fwd 算子的 API 封装：直调原生 wrapper"""

    def init_by_input_data(self, input_data: InputDataset):
        """预处理（不计时）：取 ATK 输入 -> 转目标设备 -> 校验约束 -> 确定性 seed"""
        device = get_device()
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ====== 从 ATK 获取输入（shape/dtype 已由 generate_ln_mul_dropout_fwd.py 保证约束） ======
        x_in = input_data.kwargs.get("x", None)
        u_in = input_data.kwargs.get("u", None)
        weight_in = input_data.kwargs.get("weight", None)
        bias_in = input_data.kwargs.get("bias", None)
        eps = input_data.kwargs.get("eps", None)
        dropout_ratio = input_data.kwargs.get("dropout_ratio", 0.0)
        training = input_data.kwargs.get("training", False)
        concat_ux = input_data.kwargs.get("concat_ux", False)

        if x_in is None or u_in is None:
            raise ValueError("x/u must be provided by ATK input_data")
        if weight_in is None or bias_in is None:
            raise ValueError("weight/bias must be provided by ATK input_data")
        if not isinstance(training, bool):
            try:
                training = bool(training)
            except Exception:
                raise ValueError(f"training must be bool, got {training!r}")
        if not isinstance(concat_ux, bool):
            try:
                concat_ux = bool(concat_ux)
            except Exception:
                raise ValueError(f"concat_ux must be bool, got {concat_ux!r}")

        # ====== 转换到目标设备 ======
        # 统一转成 fp32（同 golden：triton_batched_dense_vec_jagged_2d_matmul_api.py
        #   "Fix: NPU强制转为fp32，避免fp16/bf16导致kernel内类型不一致"）。
        # 原因：wrapper 内 tl.store 写回 y 时若输出为 fp16/bf16，会在昇腾 bishengir
        #   的 HFusion 阶段生成 fp32→fp16(bf16) 的 arith::TruncFOp（Ascend 工具链不支持，
        #   全部 fp32 case 编译失败：60/180)。API 全转 fp32 后，kernel 内计算与写回均
        #   为 fp32，不再产生 fp32→低精度截断，避坑且与 golden 完全一致。
        if not isinstance(x_in, torch.Tensor):
            x = torch.tensor(x_in, dtype=torch.float32, device=device)
        else:
            x = x_in.to(device=device, dtype=torch.float32)
        if not isinstance(u_in, torch.Tensor):
            u = torch.tensor(u_in, dtype=torch.float32, device=device)
        else:
            u = u_in.to(device=device, dtype=torch.float32)
        if not isinstance(weight_in, torch.Tensor):
            weight = torch.tensor(weight_in, dtype=torch.float32, device=device)
        else:
            weight = weight_in.to(device=device, dtype=torch.float32)
        if not isinstance(bias_in, torch.Tensor):
            bias = torch.tensor(bias_in, dtype=torch.float32, device=device)
        else:
            bias = bias_in.to(device=device, dtype=torch.float32)

        dtype_name = str(x.dtype).replace('torch.', '')

        # ====== 维度校验（约束由 Generator 保证；解析失败则抛出，由 atk 定位） ======
        if x.ndim != 2:
            raise ValueError(f"x must be 2D [N, D], got {x.ndim}D")
        N, D = x.shape
        if N == 0:
            raise ValueError("x.shape[0] must be > 0 (N==0 是 wrapper 空输入分支，不测)")
        if weight.ndim != 1 or bias.ndim != 1:
            raise ValueError("weight/bias must be 1D [D]")
        if weight.numel() != D or bias.numel() != D:
            raise ValueError(
                f"weight/bias numel must equal D={D}, got {weight.numel()}/{bias.numel()}"
            )

        # u：2D [N, D] 或 3D [N, H, UD]（H*UD==D）
        if u.ndim == 2:
            if u.shape != (N, D):
                raise ValueError(f"u(2D) must be [N, D]={[N, D]}, got {list(u.shape)}")
        elif u.ndim == 3:
            UN, UH, UD = u.shape
            if UN != N or UH * UD != D or UH <= 0 or UD <= 0:
                raise ValueError(
                    f"u(3D) must be [N, H, UD] with H*UD==D={D}, got {list(u.shape)}"
                )
        else:
            raise ValueError(f"u must be 2D or 3D, got {u.ndim}D")

        # BLOCK_D 上限校验（BLOCK_D = min(65536//elem_size, next_pow2(D))）
        elem_size = _ELEMENT_SIZE.get(dtype_name, 4)
        max_d = 65536 // elem_size
        if D > max_d:
            raise ValueError(
                f"D={D} exceeds BLOCK_D max {max_d} for {dtype_name}; "
                "Generator should keep D <= 65536//elem_size"
            )

        # ====== 确定性 seed ======
        import triton
        block_d = min(max_d, triton.next_power_of_2(D))
        seed = _derive_seed(block_d)

        # ====== 保存实例变量 ======
        self._device = device
        self._x = x
        self._u = u
        self._weight = weight
        self._bias = bias
        self._eps = float(eps if eps is not None else 1e-5)
        self._dropout_ratio = float(dropout_ratio)
        self._training = training
        self._concat_ux = concat_ux
        self._seed = seed

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """执行（计时）：只同步 + 调用原生 wrapper，返回真正输出 y"""
        device = self._device
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        '''
        print('----------------------------------look at me----------------------------------!')
        with open('./my_test.txt','a',encoding='utf-8') as f:
            f.write(str(self._x.shape))
            f.write(str(self._x.tolist()))
            f.write(str(self._u.shape))
            f.write(str(self._u.tolist()))
            f.write(str(self._weight.shape))
            f.write(str(self._weight))
            f.write(str(self._bias.shape))
            f.write(str(self._bias))
            f.write(str(self._eps))
            f.write(str(self._dropout_ratio))
            f.write(str(self._training))
            f.write(str(self._concat_ux))
            f.write(str(self._seed))
        print('----------------------------------look at me----------------------------------!')
        '''
        
        y, _, _, _, _, _ = triton_layer_norm_mul_dropout_fwd(
            x=self._x,
            u=self._u,
            weight=self._weight,
            bias=self._bias,
            eps=self._eps,
            dropout_ratio=self._dropout_ratio,
            training=self._training,
            concat_ux=self._concat_ux,
            seed=self._seed,
        )

        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # 真正被对比的输出：y（concat_ux=False 时 [N, D]，True 时 [N, 3D]）
        return y