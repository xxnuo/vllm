# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DeepSeek V4 MTP layer RoPE selection."""

import types

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import (
    _ROPE_DICT,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.models.deepseek_v4.attention import resolve_layer_compress_ratio
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope


@pytest.fixture(autouse=True)
def _clear_rope_cache():
    _ROPE_DICT.clear()
    yield
    _ROPE_DICT.clear()


def _config(compress_ratios: list[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=4,
        compress_ratios=compress_ratios,
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )


@pytest.mark.parametrize(
    "ratios,layer_id,expected_ratio,expected_unscaled",
    [
        ([1, 4, 4, 1], 0, 1, False),
        ([1, 4, 4, 1], 1, 4, False),
        ([0, 4, 4, 1], 0, 1, False),
        ([1, 4, 4, 1, 0], 4, 1, True),
        ([1, 4, 4, 1, 4], 4, 4, False),
        ([1, 4, 4, 1], 4, 1, False),
    ],
)
def test_resolve_layer_compress_ratio(
    ratios: list[int], layer_id: int, expected_ratio: int, expected_unscaled: bool
):
    ratio, unscaled = resolve_layer_compress_ratio(_config(ratios), layer_id)
    assert (ratio, unscaled) == (expected_ratio, expected_unscaled)
    assert ratio >= 1


def test_zero_draft_ratio_uses_plain_rope():
    config = _config([1, 4, 4, 1, 0])
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
        use_unscaled_rope=True,
    )
    assert type(rope) is RotaryEmbedding
    assert not isinstance(rope, DeepseekScalingRotaryEmbedding)


def test_scaled_rope_and_config_are_unchanged():
    config = _config([1, 4, 4, 1])
    snapshot = dict(config.rope_parameters)
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
    )
    assert isinstance(rope, DeepseekV4ScalingRotaryEmbedding)
    assert config.rope_parameters == snapshot


def test_rope_cache_is_fp32_with_bf16_default():
    config = _config([1, 4, 4, 1, 0])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
            use_unscaled_rope=True,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    assert rope.cos_sin_cache.dtype == torch.float32
