"""Integration tests for libbacktrace symbol isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from isolib.model import IsolationConfig, WarningCategory
from isolib.pipeline import isolate_library
from isolib.toolchain import Toolchain

from .conftest import BuiltLibrary, build_autotools_library, download_and_extract, CACHE_DIR
from .verify import verify_autoconf_link, verify_negative_link, verify_runtime_isolation, verify_symbols

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libbacktrace-b9e40069c0b47a722286b94eb5231f7f05c08713.zip"
TARBALL_HASH = "8acc3324d367413f41a7a65d86b96ad536907ef8c54905041670f5b9adac229a45d454a954385f230c4236d0d69d486551194b1298694af9df2229a05e998997"
TARBALL_HASH_ALGO = "sha512"
SYMBOL_PATTERNS = ["backtrace_*"]
CONSUMER_FUNC = "backtrace_create_state"
LINK_NAME = "backtrace"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


@pytest.fixture(scope="module")
def libbacktrace_built() -> BuiltLibrary:
    install_dir = CACHE_DIR / "built" / "libbacktrace"

    so_path = None
    for libdir in ["lib", "lib64"]:
        candidate = install_dir / libdir / "libbacktrace.so"
        if candidate.exists():
            so_path = candidate
            break

    if so_path is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH, TARBALL_HASH_ALGO)
        try:
            build_autotools_library(
                source_dir, install_dir,
                configure_args=[
                    "--enable-shared", "--disable-static",
                    "--with-pic",
                ],
            )
        except subprocess.CalledProcessError as e:
            pytest.skip(f"libbacktrace build failed: {e.stderr[-200:] if e.stderr else str(e)}")
        for libdir in ["lib", "lib64"]:
            candidate = install_dir / libdir / "libbacktrace.so"
            if candidate.exists():
                so_path = candidate
                break

    if so_path is None or not so_path.exists():
        pytest.skip("libbacktrace .so not found after build")
    return BuiltLibrary(
        name="libbacktrace", so_path=so_path.resolve(), install_dir=install_dir,
        symbol_patterns=SYMBOL_PATTERNS, consumer_func=CONSUMER_FUNC, link_name=LINK_NAME,
    )


@pytest.fixture(scope="module")
def libbacktrace_isolated(libbacktrace_built: BuiltLibrary, toolchain: Toolchain):
    output_dir = CACHE_DIR / "isolated" / "libbacktrace"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IsolationConfig(
        input_so=libbacktrace_built.so_path, prefix="rocm_", output_dir=output_dir,
        output_name="backtrace",
        allow_categories={WarningCategory.OBJECT_SYMBOL},
    )
    result = isolate_library(config, toolchain)
    return config, result


class TestLibbacktraceSymbols:
    def test_all_symbols_prefixed(self, libbacktrace_isolated, toolchain):
        _, result = libbacktrace_isolated
        vr = verify_symbols(result, "rocm_", SYMBOL_PATTERNS, toolchain)
        assert len(vr.prefixed_symbols) > 5
        assert not vr.unprefixed_leaks, f"Leaks: {vr.unprefixed_leaks[:10]}"


class TestLibbacktraceLinking:
    def test_autoconf_link(self, libbacktrace_isolated, toolchain, tmp_path):
        _, result = libbacktrace_isolated
        assert verify_autoconf_link(result, CONSUMER_FUNC, LINK_NAME, toolchain, tmp_path)

    def test_negative_link(self, libbacktrace_isolated, toolchain, tmp_path):
        _, result = libbacktrace_isolated
        assert verify_negative_link(result, CONSUMER_FUNC, toolchain, tmp_path)


class TestLibbacktraceRuntime:
    def test_isolated_only(self, libbacktrace_isolated, toolchain, tmp_path):
        _, result = libbacktrace_isolated
        ok, bindings = verify_runtime_isolation(
            result, CONSUMER_FUNC, LINK_NAME, "rocm_", toolchain, tmp_path,
        )
        assert ok, "\n".join(bindings[-5:])
