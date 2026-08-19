# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused regression tests for the Triton MQA logits kernels."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton MQA logits kernels require CUDA/ROCm",
)


def test_fp8_paged_mqa_logits_triton_masks_unaligned_tail_store():
    batch_size, next_n, context_len = 2, 4, 130
    num_heads, head_dim, block_size = 16, 128, 64
    device = "cuda"

    q = torch.zeros(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    kv_cache = torch.zeros(
        batch_size * 3,
        block_size,
        1,
        head_dim + 4,
        dtype=torch.uint8,
        device=device,
    )
    weights = torch.ones(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )
    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    block_tables = torch.arange(batch_size * 3, dtype=torch.int32, device=device).view(
        batch_size, 3
    )

    logits = fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=context_len,
        clean_logits=False,
    )

    q_offsets = torch.arange(context_len - next_n, context_len, device=device).repeat(
        batch_size
    )
    valid = torch.arange(context_len, device=device)[None, :] <= q_offsets[:, None]
    expected = torch.where(valid, 0.0, float("-inf"))
    torch.testing.assert_close(logits, expected)
