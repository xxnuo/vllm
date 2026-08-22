# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.utils.humming as humming
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe


def test_humming_moe_requires_current_device_heuristics(monkeypatch):
    monkeypatch.setattr(fused_humming_moe, "has_humming", lambda: True)
    monkeypatch.setattr(fused_humming_moe.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        fused_humming_moe.current_platform,
        "has_device_capability",
        lambda _: True,
    )
    assert "get_heuristics_class" in humming._EXPORTS

    def missing_heuristics():
        raise KeyError(110)

    monkeypatch.setattr(
        humming, "get_heuristics_class", missing_heuristics, raising=False
    )
    assert not fused_humming_moe.HummingGroupedExperts._supports_current_device()

    monkeypatch.setattr(humming, "get_heuristics_class", lambda: object())
    assert fused_humming_moe.HummingGroupedExperts._supports_current_device()
