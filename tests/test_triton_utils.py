# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest import mock

from vllm.triton_utils import importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"


def test_ptxas_configuration_respects_existing_path(monkeypatch):
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/custom/ptxas")
    importing._configure_triton_ptxas_for_new_gpus()
    assert importing.os.environ["TRITON_PTXAS_PATH"] == "/custom/ptxas"


def test_ptxas_configuration_skips_old_arch(monkeypatch, tmp_path):
    import sys

    ptxas = tmp_path / "bin" / "ptxas"
    ptxas.parent.mkdir()
    ptxas.write_text("#!/bin/sh\nexit 0\n")
    ptxas.chmod(0o755)
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    driver = mock.Mock()
    driver.is_active.return_value = True
    driver.return_value.get_current_target.return_value.arch = 100
    triton = types.ModuleType("triton")
    triton.__path__ = []
    backends = types.ModuleType("triton.backends")
    backends.backends = {"nvidia": mock.Mock(driver=driver)}
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    importing._configure_triton_ptxas_for_new_gpus()
    assert "TRITON_PTXAS_PATH" not in importing.os.environ


def test_ptxas_configuration_sets_system_path_for_new_arch(monkeypatch, tmp_path):
    import sys

    ptxas = tmp_path / "bin" / "ptxas"
    ptxas.parent.mkdir()
    ptxas.write_text("#!/bin/sh\nexit 0\n")
    ptxas.chmod(0o755)
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    driver = mock.Mock()
    driver.is_active.return_value = True
    driver.return_value.get_current_target.return_value.arch = 110
    triton = types.ModuleType("triton")
    triton.__path__ = []
    backends = types.ModuleType("triton.backends")
    backends.backends = {"nvidia": mock.Mock(driver=driver)}
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    importing._configure_triton_ptxas_for_new_gpus()
    assert importing.os.environ["TRITON_PTXAS_PATH"] == str(ptxas)
