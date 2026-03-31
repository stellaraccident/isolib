"""Integration tests for libcap symbol isolation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from isolib.model import IsolationConfig, WarningCategory
from isolib.pipeline import isolate_library
from isolib.toolchain import Toolchain

from .conftest import BuiltLibrary, download_and_extract, CACHE_DIR
from .verify import verify_autoconf_link, verify_negative_link, verify_runtime_isolation, verify_symbols

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libcap-2.69.tar.gz"
TARBALL_HASH = "9cc2fa3ee744d881629cdac1a1b54c302e7684fda3e3622056218c7681642ffb"
SYMBOL_PATTERNS = ["cap_*", "capng_*"]
CONSUMER_FUNC = "cap_init"
LINK_NAME = "cap"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


@pytest.fixture(scope="module")
def libcap_built() -> BuiltLibrary:
    install_dir = CACHE_DIR / "built" / "libcap-2.69"

    so_path = None
    for libdir in ["lib", "lib64"]:
        candidate = install_dir / libdir / "libcap.so"
        if candidate.exists():
            so_path = candidate
            break

    if so_path is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH)
        install_dir.mkdir(parents=True, exist_ok=True)
        lib_dir = install_dir / "lib"
        lib_dir.mkdir(exist_ok=True)

        # libcap uses a plain Makefile; build libcap only (skip pam, tests)
        env = {**os.environ, "DESTDIR": "", "prefix": str(install_dir),
               "lib": "lib", "RAISE_SETFCAP": "no"}
        try:
            subprocess.run(
                ["make", "-C", "libcap", "-j", str(os.cpu_count() or 4),
                 f"prefix={install_dir}", "lib=lib", "RAISE_SETFCAP=no"],
                check=True, capture_output=True, text=True, cwd=source_dir, env=env,
            )
            subprocess.run(
                ["make", "-C", "libcap", "install",
                 f"prefix={install_dir}", "lib=lib", "RAISE_SETFCAP=no",
                 f"DESTDIR="],
                check=True, capture_output=True, text=True, cwd=source_dir, env=env,
            )
        except subprocess.CalledProcessError as e:
            pytest.skip(f"libcap build failed: {e.stderr[-200:]}")

        for libdir in ["lib", "lib64"]:
            candidate = install_dir / libdir / "libcap.so"
            if candidate.exists():
                so_path = candidate
                break

    if so_path is None or not so_path.exists():
        pytest.skip("libcap .so not found after build")
    return BuiltLibrary(
        name="libcap", so_path=so_path.resolve(), install_dir=install_dir,
        symbol_patterns=SYMBOL_PATTERNS, consumer_func=CONSUMER_FUNC, link_name=LINK_NAME,
    )


@pytest.fixture(scope="module")
def libcap_isolated(libcap_built: BuiltLibrary, toolchain: Toolchain):
    output_dir = CACHE_DIR / "isolated" / "libcap-2.69"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IsolationConfig(
        input_so=libcap_built.so_path, prefix="rocm_", output_dir=output_dir,
        output_name="cap", allow_categories={WarningCategory.OBJECT_SYMBOL},
    )
    result = isolate_library(config, toolchain)
    return config, result


class TestLibcapSymbols:
    def test_all_symbols_prefixed(self, libcap_isolated, toolchain):
        _, result = libcap_isolated
        vr = verify_symbols(result, "rocm_", SYMBOL_PATTERNS, toolchain)
        assert len(vr.prefixed_symbols) > 30
        assert not vr.unprefixed_leaks, f"Leaks: {vr.unprefixed_leaks[:10]}"


class TestLibcapLinking:
    def test_autoconf_link(self, libcap_isolated, toolchain, tmp_path):
        _, result = libcap_isolated
        assert verify_autoconf_link(result, CONSUMER_FUNC, LINK_NAME, toolchain, tmp_path)

    def test_negative_link(self, libcap_isolated, toolchain, tmp_path):
        _, result = libcap_isolated
        assert verify_negative_link(result, CONSUMER_FUNC, toolchain, tmp_path)


class TestLibcapRuntime:
    def test_isolated_only(self, libcap_isolated, toolchain, tmp_path):
        _, result = libcap_isolated
        ok, bindings = verify_runtime_isolation(
            result, CONSUMER_FUNC, LINK_NAME, "rocm_", toolchain, tmp_path,
        )
        assert ok, "\n".join(bindings[-5:])
