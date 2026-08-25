# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.compilation.backends as compilation_backends
from vllm.config import CompilationMode
from vllm.model_executor.models import qwen3_dflash, qwen3_dflash2
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


def test_dflash2_loader_aliases_target_vocab_modules(monkeypatch):
    target_embed = object()
    target_lm_head = object()
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_lm_head,
    )
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            attention_backend=None,
            kv_cache_dtype=None,
        ),
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
    )

    monkeypatch.setattr(dflash_utils, "get_model", lambda **_: draft_model)
    monkeypatch.setattr(
        dflash_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(dflash_utils, "replace", lambda obj, **_: obj)
    monkeypatch.setattr(compilation_backends, "set_model_tag", lambda _: nullcontext())
    monkeypatch.setattr(qwen3_dflash, "dflash_has_any_non_causal", lambda _: False)
    monkeypatch.setattr(
        qwen3_dflash, "dflash_target_rope_is_neox_style", lambda _: None
    )

    loaded_draft = dflash_utils.load_dflash_model(target_model, vllm_config)

    assert loaded_draft.model.embed_tokens is target_embed
    assert loaded_draft.lm_head is target_lm_head


def test_dflash2_rejects_pipeline_parallelism():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2)
    )

    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        DFlash2Qwen3ForCausalLM(vllm_config=vllm_config)


def test_dflash2_constructs_decoder_layers_and_vocab_placeholders(monkeypatch):
    layer_indices = []

    class VocabModule(nn.Module):
        def __init__(self, vocab_size, *_args, **_kwargs):
            super().__init__()
            self.vocab_size = vocab_size

    class NoopModule(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    def init_dflash2_layer(self, _vllm_config, *, layer_idx, **_kwargs):
        nn.Module.__init__(self)
        layer_indices.append(layer_idx)

    def init_base_layer(*_args, **_kwargs):
        pytest.fail("DFlash2 must not construct DFlashQwen3DecoderLayer")

    hf_config = SimpleNamespace(
        vocab_size=1024,
        hidden_size=16,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        eagle_config={},
        dflash_config={
            "selector_rank": 2,
            "selector_top_k": 2,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            get_num_layers=lambda _: 4,
            get_vocab_size=lambda: 1024,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    monkeypatch.setattr(qwen3_dflash, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(qwen3_dflash, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(qwen3_dflash, "VocabParallelEmbedding", VocabModule)
    monkeypatch.setattr(qwen3_dflash, "ParallelLMHead", VocabModule)
    monkeypatch.setattr(qwen3_dflash, "ReplicatedLinear", NoopModule)
    monkeypatch.setattr(qwen3_dflash, "RMSNorm", NoopModule)
    monkeypatch.setattr(qwen3_dflash, "LogitsProcessor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qwen3_dflash2, "CandidateSelector", NoopModule)
    monkeypatch.setattr(
        qwen3_dflash2,
        "LogitsProcessor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(qwen3_dflash2, "set_model_tag", lambda _: nullcontext())
    monkeypatch.setattr(
        qwen3_dflash.DFlashQwen3DecoderLayer, "__init__", init_base_layer
    )
    monkeypatch.setattr(
        qwen3_dflash2.DFlash2Qwen3DecoderLayer,
        "__init__",
        init_dflash2_layer,
    )

    model = DFlash2Qwen3ForCausalLM(vllm_config=vllm_config)

    assert layer_indices == [0, 1]
    assert all(
        isinstance(layer, qwen3_dflash2.DFlash2Qwen3DecoderLayer)
        for layer in model.model.layers
    )
    assert model.model.embed_tokens.vocab_size == 1
    assert model.lm_head.vocab_size == 1
