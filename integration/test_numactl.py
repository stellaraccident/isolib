"""Integration tests for numactl (libnuma) symbol isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from isolib.model import IsolationConfig, WarningCategory
from isolib.pipeline import isolate_library
from isolib.toolchain import Toolchain

from .conftest import BuiltLibrary, build_autotools_library, download_and_extract, CACHE_DIR
from .verify import verify_autoconf_link, verify_negative_link, verify_runtime_isolation, verify_symbols

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/numactl-2.0.19.tar.gz"
TARBALL_HASH = "f2672a0381cb59196e9c246bf8bcc43d5568bc457700a697f1a1df762b9af884"
SYMBOL_PATTERNS = ["numa_*", "nodemask_*"]
CONSUMER_FUNC = "numa_available"
LINK_NAME = "numa"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


@pytest.fixture(scope="module")
def numactl_built() -> BuiltLibrary:
    install_dir = CACHE_DIR / "built" / "numactl-2.0.19"

    so_path = None
    for libdir in ["lib", "lib64"]:
        candidate = install_dir / libdir / "libnuma.so"
        if candidate.exists():
            so_path = candidate
            break

    if so_path is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH)
        try:
            build_autotools_library(
                source_dir, install_dir,
                configure_args=["--enable-shared", "--disable-static"],
            )
        except subprocess.CalledProcessError:
            # numactl may need autoreconf
            subprocess.run(
                ["autoreconf", "-fi"], cwd=source_dir,
                check=True, capture_output=True, text=True,
            )
            build_autotools_library(
                source_dir, install_dir,
                configure_args=["--enable-shared", "--disable-static"],
            )
        for libdir in ["lib", "lib64"]:
            candidate = install_dir / libdir / "libnuma.so"
            if candidate.exists():
                so_path = candidate
                break

    if so_path is None or not so_path.exists():
        pytest.skip("libnuma .so not found after build")
    return BuiltLibrary(
        name="numactl", so_path=so_path.resolve(), install_dir=install_dir,
        symbol_patterns=SYMBOL_PATTERNS, consumer_func=CONSUMER_FUNC, link_name=LINK_NAME,
    )


@pytest.fixture(scope="module")
def numactl_isolated(numactl_built: BuiltLibrary, toolchain: Toolchain):
    output_dir = CACHE_DIR / "isolated" / "numactl-2.0.19"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IsolationConfig(
        input_so=numactl_built.so_path, prefix="rocm_", output_dir=output_dir,
        output_name="numa",
        allow_categories={WarningCategory.OBJECT_SYMBOL, WarningCategory.VERSIONED_SYMBOL},
    )
    result = isolate_library(config, toolchain)
    return config, result


class TestNumactlSymbols:
    def test_all_symbols_prefixed(self, numactl_isolated, toolchain):
        _, result = numactl_isolated
        vr = verify_symbols(result, "rocm_", SYMBOL_PATTERNS, toolchain)
        assert len(vr.prefixed_symbols) > 20
        assert not vr.unprefixed_leaks, f"Leaks: {vr.unprefixed_leaks[:10]}"


class TestNumactlLinking:
    def test_autoconf_link(self, numactl_isolated, toolchain, tmp_path):
        _, result = numactl_isolated
        assert verify_autoconf_link(result, CONSUMER_FUNC, LINK_NAME, toolchain, tmp_path)

    def test_negative_link(self, numactl_isolated, toolchain, tmp_path):
        _, result = numactl_isolated
        assert verify_negative_link(result, CONSUMER_FUNC, toolchain, tmp_path)


class TestNumactlRuntime:
    def test_isolated_only(self, numactl_isolated, toolchain, tmp_path):
        _, result = numactl_isolated
        ok, bindings = verify_runtime_isolation(
            result, CONSUMER_FUNC, LINK_NAME, "rocm_", toolchain, tmp_path,
        )
        assert ok, "\n".join(bindings[-5:])
