# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.platforms.interface import DeviceCapability


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel


def _thor_deepseek_v4_moe_config():
    return SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
        moe_backend="auto",
        routing_method=RoutingMethodType.DeepseekV4,
    )


def test_deepseek_v4_thor_mxfp4_falls_back_to_marlin(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4

    class UnsupportedExperts:
        @staticmethod
        def is_supported_config(*args):
            return False, "triton_kernels is unavailable"

    class SupportedExperts:
        @staticmethod
        def is_supported_config(*args):
            return True, None

    attempted = []

    def kernel_classes(backend):
        attempted.append(backend)
        if backend == Mxfp4MoeBackend.TRITON_UNFUSED:
            return [UnsupportedExperts]
        return [SupportedExperts]

    monkeypatch.setattr(mxfp4.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        mxfp4.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(11, 0),
    )
    monkeypatch.setattr(mxfp4, "backend_to_kernel_cls", kernel_classes)

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(
        _thor_deepseek_v4_moe_config()
    )

    assert backend == Mxfp4MoeBackend.MARLIN
    assert experts_cls is SupportedExperts
    assert attempted == [Mxfp4MoeBackend.TRITON_UNFUSED, Mxfp4MoeBackend.MARLIN]


def test_deepseek_v4_thor_mxfp4_skips_unfused_triton_for_mtp(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4

    class SupportedExperts:
        @staticmethod
        def is_supported_config(*args):
            return True, None

    attempted = []

    def kernel_classes(backend):
        attempted.append(backend)
        return [SupportedExperts]

    monkeypatch.setattr(mxfp4.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        mxfp4.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(11, 0),
    )
    monkeypatch.setattr(mxfp4, "backend_to_kernel_cls", kernel_classes)

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(
        _thor_deepseek_v4_moe_config(),
        allow_auto_triton_unfused=False,
    )

    assert backend == Mxfp4MoeBackend.MARLIN
    assert experts_cls is SupportedExperts
    assert attempted == [Mxfp4MoeBackend.MARLIN]


def test_deepseek_v4_mxfp4_config_marks_mtp_prefix(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.models.deepseek_v4 import quant_config as deepseek_v4_quant

    captured = []

    class DummyMethod:
        def __init__(self, moe, *, allow_auto_triton_unfused=True):
            captured.append(allow_auto_triton_unfused)

    model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: 61)
    monkeypatch.setattr(
        deepseek_v4_quant,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    monkeypatch.setattr(deepseek_v4_quant, "Mxfp4MoEMethod", DummyMethod)

    config = object.__new__(deepseek_v4_quant.DeepseekV4FP8Config)
    config._resolved_expert_dtype = "fp4"
    config._resolved_moe_quant_algo = ""
    config.ignored_layers = []
    config.packed_modules_mapping = {}
    layer = object.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(routing_method=RoutingMethodType.DeepseekV4)

    config.get_quant_method(layer, "model.layers.60.ffn.experts")
    config.get_quant_method(layer, "model.layers.61.ffn.experts")

    assert captured == [True, False]
