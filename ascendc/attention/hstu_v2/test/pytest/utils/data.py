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
# pylint: disable=duplicate-code
import random

import numpy as np
import torch
import torch_npu

from .mask import create_causal_mask


class TestDataGenerator:
    def __init__(self, seed, seq_all_equal, seq_max_ratio=1):
        self.seed = seed
        self.seq_all_equal = seq_all_equal
        self.seq_max_ratio = seq_max_ratio
        self.__init_seed()

    @staticmethod
    def __gen_random_sequence(batch_size, max_seqlen_q, max_seqlen_k, num_context, num_target):
        min_seqlen = 1
        if num_context is not None:
            min_seqlen += num_context
        if num_target is not None:
            min_seqlen += num_target
        seq_lens_q = torch.randint(min_seqlen, max_seqlen_q + 1, (batch_size,), dtype=torch.int32)
        seq_lens_k = torch.randint(min_seqlen, max_seqlen_k + 1, (batch_size,), dtype=torch.int32)
        seq_lens_q = torch.where(seq_lens_k < seq_lens_q, seq_lens_k, seq_lens_q)
        return seq_lens_q, seq_lens_k

    def gen_data(
        self,
        batch_size,
        head_num,
        max_seqlen_q,
        max_seqlen_k,
        head_dim_qk,
        head_dim_v,
        has_rab,
        data_type,
        window_size=(-1, -1),
        num_context=None,
        num_target=None,
        target_group_size=None,
        seq_lens_q_override=None,
        seq_lens_k_override=None,
        rab_heads=None,
    ):
        if seq_lens_q_override is not None or seq_lens_k_override is not None:
            if seq_lens_q_override is None or seq_lens_k_override is None:
                raise ValueError("seq_lens_q_override and seq_lens_k_override must be specified together")
            if len(seq_lens_q_override) != batch_size or len(seq_lens_k_override) != batch_size:
                raise ValueError("sequence-length override size must match batch_size")
            seq_lens_q = torch.as_tensor(seq_lens_q_override, dtype=torch.int32)
            seq_lens_k = torch.as_tensor(seq_lens_k_override, dtype=torch.int32)
            if torch.any(seq_lens_q < 1) or torch.any(seq_lens_q > max_seqlen_q):
                raise ValueError("query sequence-length override is out of range")
            if torch.any(seq_lens_k < 1) or torch.any(seq_lens_k > max_seqlen_k):
                raise ValueError("key sequence-length override is out of range")
            if torch.any(seq_lens_q > seq_lens_k):
                raise ValueError("HSTU requires every query sequence length to be <= its key length")
        else:
            seq_lens_q, seq_lens_k = self.__gen_sequence(
                batch_size, max_seqlen_q, max_seqlen_k, num_context, num_target
            )

        seq_offset_q = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_q, axis=0))).numpy()
        seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_k, axis=0))).numpy()

        total_len_q = torch.sum(seq_lens_q).item()
        total_len_k = torch.sum(seq_lens_k).item()

        grad = torch.rand(total_len_q, head_num, head_dim_v, dtype=data_type).uniform_(-1, 1)
        q = torch.rand(total_len_q, head_num, head_dim_qk, dtype=data_type).uniform_(-1, 1)
        k = torch.rand(total_len_k, head_num, head_dim_qk, dtype=data_type).uniform_(-1, 1)
        v = torch.rand(total_len_k, head_num, head_dim_v, dtype=data_type).uniform_(-1, 1)

        if has_rab:
            actual_rab_heads = head_num if rab_heads is None else rab_heads
            if actual_rab_heads not in (1, head_num):
                raise ValueError("rab_heads must be 1 or head_num")
            rab = torch.rand(batch_size, actual_rab_heads, max_seqlen_q, max_seqlen_k, dtype=data_type).uniform_(-1, 1)
        else:
            rab = None

        if window_size == (-1, 0):
            mask = torch.zeros(batch_size, head_num, max_seqlen_q, max_seqlen_k)
            for sample_id, (seq_len_q, seq_len_k) in enumerate(zip(seq_lens_q, seq_lens_k)):
                mask_tensor = create_causal_mask(seq_len_q, seq_len_k, num_context, num_target, target_group_size)
                mask[sample_id, :, :seq_len_q, :seq_len_k] = mask_tensor
            mask = mask.to(data_type)
        elif window_size == (-1, -1):
            mask = None
        else:
            mask = torch.randint(0, 2, (batch_size, head_num, max_seqlen_q, max_seqlen_k), dtype=data_type)
        return grad, q, k, v, rab, mask, seq_offset_q, seq_offset_k

    def __init_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch_npu.npu.manual_seed_all(self.seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
        torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化

    def __gen_all_equal_sequence(self, batch_size, max_seqlen_q, max_seqlen_k):
        average_max_seqlen_q = int(self.seq_max_ratio * max_seqlen_q)
        average_max_seqlen_k = int(self.seq_max_ratio * max_seqlen_k)
        seq_lens_q = torch.randint(average_max_seqlen_q, average_max_seqlen_q + 1, (batch_size,), dtype=torch.int32)
        seq_lens_k = torch.randint(average_max_seqlen_k, average_max_seqlen_k + 1, (batch_size,), dtype=torch.int32)
        return seq_lens_q, seq_lens_k

    def __gen_sequence(self, batch_size, max_seqlen_q, max_seqlen_k, num_context, num_target):
        if self.seq_all_equal:
            seq_lens_q, seq_lens_k = self.__gen_all_equal_sequence(batch_size, max_seqlen_q, max_seqlen_k)
        else:
            seq_lens_q, seq_lens_k = self.__gen_random_sequence(
                batch_size, max_seqlen_q, max_seqlen_k, num_context, num_target
            )
        return seq_lens_q, seq_lens_k
