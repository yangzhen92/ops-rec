#!/usr/bin/env python3
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
import ops_rec


def test_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    torch.npu.set_device(0)

    # 准备输入数据
    b, n, s, d = 2, 4, 256, 32
    query_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    key_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    value_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    pos_key_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    pos_query_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    relative_pos = torch.randint(0, 10, (1, 1, s, s), device='npu', dtype=torch.int64)
    attn_mask = torch.randn(b, 1, s, s, device='npu', dtype=torch.float16)

    # 调用算子
    outputs, probs, weights = ops_rec.ascendc.attention.disentangle_attention(
        query_layer,
        key_layer,
        value_layer,
        pos_key_layer,
        pos_query_layer,
        relative_pos,
        attn_mask,
        pos_attr_type="c2p|p2c",
        score_scale=1.0,
    )

    print(f"Query layer shape: {query_layer.shape}")
    print(f"Output shape: {outputs.shape}")  # [2, 4, 256, 32]
    print(f"Probs shape: {probs.shape}")  # [2, 4, 256, 256]
    print(f"Weights shape: {weights.shape}")  # [2, 4, 256, 256]
    print(f"Output dtype: {outputs.dtype}")
    print()


def test_different_modes():
    """测试不同的位置编码模式"""
    print("=== 不同位置编码模式示例 ===")
    torch.npu.set_device(0)

    b, n, s, d = 1, 2, 256, 16
    query_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    key_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    value_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    pos_key_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    pos_query_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    relative_pos = torch.randint(0, 10, (1, 1, s, s), device='npu', dtype=torch.int64)
    attn_mask = torch.randn(b, 1, s, s, device='npu', dtype=torch.float16)

    # c2p 模式：当前到过去的注意力
    print("c2p 模式:")
    outputs_c2p, _, _ = ops_rec.ascendc.attention.disentangle_attention(
        query_layer,
        key_layer,
        value_layer,
        pos_key_layer,
        pos_query_layer,
        relative_pos,
        attn_mask,
        pos_attr_type="c2p",
        score_scale=1.0,
    )
    print(f"  Output shape: {outputs_c2p.shape}")

    # p2c 模式：过去到当前的注意力
    print("p2c 模式:")
    outputs_p2c, _, _ = ops_rec.ascendc.attention.disentangle_attention(
        query_layer,
        key_layer,
        value_layer,
        pos_key_layer,
        pos_query_layer,
        relative_pos,
        attn_mask,
        pos_attr_type="p2c",
        score_scale=1.0,
    )
    print(f"  Output shape: {outputs_p2c.shape}")

    # c2p|p2c 模式：双向注意力
    print("c2p|p2c 模式:")
    outputs_both, _, _ = ops_rec.ascendc.attention.disentangle_attention(
        query_layer,
        key_layer,
        value_layer,
        pos_key_layer,
        pos_query_layer,
        relative_pos,
        attn_mask,
        pos_attr_type="c2p|p2c",
        score_scale=1.0,
    )
    print(f"  Output shape: {outputs_both.shape}")
    print()


def test_compile_mode():
    """torch.compile 图模式示例"""
    print("=== torch.compile 图模式示例 ===")
    torch.npu.set_device(0)

    # 定义模型
    def model(query, key, value, pos_key, pos_query, rel_pos, mask):
        return ops_rec.ascendc.attention.disentangle_attention(
            query, key, value, pos_key, pos_query, rel_pos, mask, pos_attr_type="c2p|p2c", score_scale=1.0
        )

    # 编译模型
    compiled_model = torch.compile(model, dynamic=True, fullgraph=True)

    # 准备输入数据
    b, n, s, d = 1, 2, 256, 16
    query_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    key_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    value_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    pos_key_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    pos_query_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    relative_pos = torch.randint(0, 10, (1, 1, s, s), device='npu', dtype=torch.int64)
    attn_mask = torch.randn(b, 1, s, s, device='npu', dtype=torch.float16)

    # 调用编译后的模型
    outputs, probs, weights = compiled_model(
        query_layer, key_layer, value_layer, pos_key_layer, pos_query_layer, relative_pos, attn_mask
    )

    print(f"Compiled model output shape: {outputs.shape}")
    print(f"Compiled model probs shape: {probs.shape}")
    print(f"Compiled model weights shape: {weights.shape}")
    print()


def test_score_scale():
    """测试不同的分数缩放因子"""
    print("=== 分数缩放因子示例 ===")
    torch.npu.set_device(0)

    b, n, s, d = 1, 2, 256, 16
    query_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    key_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    value_layer = torch.randn(b, n, s, d, device='npu', dtype=torch.float16)
    pos_key_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    pos_query_layer = torch.randn(2 * s, n, d, device='npu', dtype=torch.float16)
    relative_pos = torch.randint(0, 10, (1, 1, s, s), device='npu', dtype=torch.int64)
    attn_mask = torch.randn(b, 1, s, s, device='npu', dtype=torch.float16)

    for scale in [0.5, 1.0, 2.0]:
        outputs, _, _ = ops_rec.ascendc.attention.disentangle_attention(
            query_layer,
            key_layer,
            value_layer,
            pos_key_layer,
            pos_query_layer,
            relative_pos,
            attn_mask,
            pos_attr_type="c2p|p2c",
            score_scale=scale,
        )
        print(f"Score scale {scale}: output mean = {outputs.mean().item():.4f}")
    print()


def main():
    """运行所有示例"""
    try:
        test_basic_usage()
        test_different_modes()
        test_compile_mode()
        test_score_scale()
        print("所有示例执行完成！")
    except Exception as e:
        print(f"执行示例时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
