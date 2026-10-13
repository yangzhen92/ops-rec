#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import ops_rec  # noqa: F401 - initializes packaged custom OPP paths

torch.npu.config.allow_internal_format = False


class Kernel:
    def __init__(self, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        self.alpha = alpha
        self.scale = scale
        self.has_rab = has_rab
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

        self.seq_offset_q = seq_offset_q
        self.seq_offset_k = seq_offset_k

    def forward(
        self,
        q,
        k,
        v,
        rab,
        mask,
        metadata=None,
        sparse_info=None,
        num_context=None,
        num_target=None,
        target_group_size=None,
        window_size=(-1, -1),
        deterministic=True,
        plan_af=None,
    ):
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        rab_npu = rab.to("npu") if isinstance(rab, torch.Tensor) else None

        seq_offset_q = torch.Tensor(self.seq_offset_q).to("npu").to(torch.int32)
        seq_offset_k = torch.Tensor(self.seq_offset_k).to("npu").to(torch.int32)

        if num_context is not None:
            num_context = torch.as_tensor(num_context, dtype=torch.int32, device="npu")
        if num_target is not None:
            num_target = torch.as_tensor(num_target, dtype=torch.int32, device="npu")
        mask_npu = None
        # No-mask calls have neither built-in mask parameters nor packed
        # sparse information.  All other test pipelines describe a causal or
        # arbitrary mask and must retain mask_type=0.
        is_local = window_size not in ((-1, -1), (-1, 0))
        mask_type = 2 if sparse_info is None and num_context is None and num_target is None and not is_local else 0

        output = torch.ops.ops_rec.hstu_forward_v2(
            q_npu,
            k_npu,
            v_npu,
            mask_npu,
            rab_npu,
            mask_type,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.scale,
            seq_offset_q,
            seq_offset_k,
            num_context,
            num_target,
            target_group_size,
            self.alpha,
            metadata,
            sparse_info,
            window_size[0],
            window_size[1],
            plan_af,
            deterministic,
        )
        torch.npu.synchronize()
        return output.cpu()

    def create_forward_metadata(self, q, k, v, sparse_info=None, tile=None, deterministic=True):
        """调用 create_attn_metadata，为 HSTU forward 的 Q 行调度生成 metadata。"""
        seq_offset_q = torch.as_tensor(self.seq_offset_q, dtype=torch.int32, device="npu")
        seq_offset_k = torch.as_tensor(self.seq_offset_k, dtype=torch.int32, device="npu")
        batch_size = seq_offset_q.numel() - 1
        head_num = q.shape[1]
        head_dim = max(q.shape[-1], v.shape[-1])
        q_block_size, kv_block_size, _ = (
            tile if tile is not None else torch.ops.ops_rec.get_fwd_tile_size(q, k, v, True)
        )

        metadata = torch.ops.ops_rec.create_attn_metadata(
            seq_offset_q,
            seq_offset_k,
            None,
            None,
            batch_size,
            self.max_seqlen_q,
            self.max_seqlen_k,
            head_num,
            head_num,
            head_dim,
            0,
            -1,
            -1,
            "TND",
            "TND",
            "TND",
            sparse_info[0] if sparse_info is not None else None,
            sparse_info[3] if sparse_info is not None else None,
            0,
            sparse_info[0].shape[1] if sparse_info is not None else 1,
            q_block_size,
            kv_block_size,
            deterministic,
        )
        torch.npu.synchronize()
        return metadata

    def create_plan(
        self,
        direction,
        q,
        k,
        v,
        af=None,
        num_context=None,
        num_target=None,
        target_group_size=1,
        window_size=(-1, -1),
        mask_type=2,
        deterministic=True,
    ):
        """Build the unified host-side tile, packed-mask and scheduling plan."""
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        af_npu = af.to("npu") if isinstance(af, torch.Tensor) else None
        seq_offset_q = torch.as_tensor(self.seq_offset_q, dtype=torch.int32, device="npu")
        seq_offset_k = torch.as_tensor(self.seq_offset_k, dtype=torch.int32, device="npu")
        batch_size = seq_offset_q.numel() - 1
        if num_context is not None:
            num_context = torch.as_tensor(num_context, dtype=torch.int32, device="npu")
            if num_context.ndim == 0:
                num_context = num_context.repeat(batch_size)
        if num_target is not None:
            num_target = torch.as_tensor(num_target, dtype=torch.int32, device="npu")
            if num_target.ndim == 0:
                num_target = num_target.repeat(batch_size)
        plan_op = torch.ops.ops_rec.get_fwd_attn_plan if direction == "forward" else torch.ops.ops_rec.get_bwd_attn_plan
        tile, sparse, metadata, use_aicpu = plan_op(
            q_npu,
            k_npu,
            v_npu,
            af_npu,
            seq_offset_q,
            seq_offset_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
            num_context,
            num_target,
            target_group_size,
            window_size[0],
            window_size[1],
            mask_type,
            deterministic,
        )
        fwd_tile = tile if direction == "forward" else list(torch.ops.ops_rec.get_fwd_tile_size(q, k, v, True))
        bwd_tile = tile if direction == "backward" else list(torch.ops.ops_rec.get_bwd_tile_size(q, k, v, True))
        return fwd_tile, bwd_tile, sparse, metadata, use_aicpu

    def create_backward_metadata(self, q, k, v, sparse_info=None, tile=None, deterministic=True):
        """调用 create_attn_metadata，为 HSTU backward 的 K 行调度生成 metadata。"""
        seq_offset_q = torch.as_tensor(self.seq_offset_q, dtype=torch.int32, device="npu")
        seq_offset_k = torch.as_tensor(self.seq_offset_k, dtype=torch.int32, device="npu")
        batch_size = seq_offset_q.numel() - 1
        head_num = q.shape[1]
        head_dim = max(q.shape[-1], v.shape[-1])
        q_block_size, kv_block_size, _ = (
            tile if tile is not None else torch.ops.ops_rec.get_bwd_tile_size(q, k, v, True)
        )

        metadata = torch.ops.ops_rec.create_attn_metadata(
            seq_offset_q,
            seq_offset_k,
            None,
            None,
            batch_size,
            self.max_seqlen_q,
            self.max_seqlen_k,
            head_num,
            head_num,
            head_dim,
            0,
            -1,
            -1,
            "TND",
            "TND",
            "TND",
            sparse_info[0] if sparse_info is not None else None,
            sparse_info[3] if sparse_info is not None else None,
            1,
            sparse_info[0].shape[1] if sparse_info is not None else 1,
            q_block_size,
            kv_block_size,
            deterministic,
        )
        torch.npu.synchronize()
        return metadata

    def backward(
        self,
        grad,
        q,
        k,
        v,
        rab=None,
        mask=None,
        window_size=(-1, -1),
        num_context=None,
        num_target=None,
        target_group_size=None,
        metadata=None,
        sparse_info=None,
        deterministic=True,
    ):
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")

        if rab is not None:
            rab_npu = rab.to("npu")
        else:
            rab_npu = None

        seq_offset_q = torch.Tensor(self.seq_offset_q).to("npu").to(torch.int32)
        seq_offset_k = torch.Tensor(self.seq_offset_k).to("npu").to(torch.int32)

        batch_size = len(seq_offset_q) - 1
        if num_context is not None:
            num_context = torch.as_tensor(num_context, dtype=torch.int32, device="npu")
            if num_context.ndim == 0:
                num_context = num_context.repeat(batch_size)
        if num_target is not None:
            num_target = torch.as_tensor(num_target, dtype=torch.int32, device="npu")
            if num_target.ndim == 0:
                num_target = num_target.repeat(batch_size)
        if sparse_info is not None:
            sparse_info = [sp.to("npu").to(torch.int32) for sp in sparse_info]
        q_grad, k_grad, v_grad, rab_grad = torch.ops.ops_rec.hstu_backward_v2(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            self.max_seqlen_q,
            self.max_seqlen_k,
            seq_offset_q,
            seq_offset_k,
            rab_npu,
            num_context,
            num_target,
            self.scale,
            target_group_size,
            self.alpha,
            window_size[0],
            window_size[1],
            metadata,
            sparse_info,
            deterministic,
        )

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), rab_grad.cpu() if rab is not None else None


class AscendFuse:
    def __init__(self, device, ops_library_dir):
        torch.npu.set_device(device)
        torch.ops.load_library(ops_library_dir)

    @staticmethod
    def kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        return Kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k)

    @staticmethod
    def validator():
        raise RuntimeError("backend ascend fuse no validator")
