"""Integration tests for libdrm symbol isolation (meson build)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from isolib.model import IsolationConfig, WarningCategory
from isolib.pipeline import isolate_library
from isolib.toolchain import Toolchain

from .conftest import BuiltLibrary, download_and_extract, CACHE_DIR
from .verify import verify_autoconf_link, verify_negative_link, verify_runtime_isolation, verify_symbols

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libdrm-libdrm-2.4.127.tar.bz2"
TARBALL_HASH = "3A11654905C595FCA2825D6E259A8086D04A448DCE215B4CB03A49320D9259EC"
SYMBOL_PATTERNS = ["drmMode*", "drmGet*", "drmOpen*", "drmClose*", "drmIoctl",
                   "drmFree*", "drmMap*", "drmUnmap*", "drmCommand*",
                   "drmAvailable", "drmHandle*", "drmSet*", "drmDrop*",
                   "drmAuth*", "drmPrime*", "drmCrtc*", "drmSyncobjCreate"]
CONSUMER_FUNC = "drmAvailable"
LINK_NAME = "drm"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


@pytest.fixture(scope="module")
def libdrm_built() -> BuiltLibrary:
    install_dir = CACHE_DIR / "built" / "libdrm-2.4.127"

    so_path = None
    for libdir in ["lib", "lib64", "lib/x86_64-linux-gnu"]:
        candidate = install_dir / libdir / "libdrm.so"
        if candidate.exists():
            so_path = candidate
            break

    if so_path is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH)
        build_dir = source_dir / "_build"

        try:
            subprocess.run(
                [
                    "meson", "setup", str(build_dir),
                    f"--prefix={install_dir}",
                    "-Damdgpu=enabled",
                    "-Dtests=false",
                    "-Dman-pages=disabled",
                    "-Dvalgrind=disabled",
                    "-Dcairo-tests=disabled",
                ],
                check=True, capture_output=True, text=True, cwd=source_dir,
            )
            subprocess.run(
                ["ninja", "-C", str(build_dir)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["ninja", "-C", str(build_dir), "install"],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pytest.skip(f"libdrm build failed (needs meson): {e}")

        for libdir in ["lib", "lib64", "lib/x86_64-linux-gnu"]:
            candidate = install_dir / libdir / "libdrm.so"
            if candidate.exists():
                so_path = candidate
                break

    if so_path is None or not so_path.exists():
        pytest.skip("libdrm .so not found after build")
    return BuiltLibrary(
        name="libdrm", so_path=so_path.resolve(), install_dir=install_dir,
        symbol_patterns=SYMBOL_PATTERNS, consumer_func=CONSUMER_FUNC, link_name=LINK_NAME,
    )


@pytest.fixture(scope="module")
def libdrm_isolated(libdrm_built: BuiltLibrary, toolchain: Toolchain):
    output_dir = CACHE_DIR / "isolated" / "libdrm-2.4.127"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IsolationConfig(
        input_so=libdrm_built.so_path, prefix="rocm_", output_dir=output_dir,
        output_name="drm",
        allow_categories={WarningCategory.OBJECT_SYMBOL, WarningCategory.VERSIONED_SYMBOL},
    )
    result = isolate_library(config, toolchain)
    return config, result


class TestLibdrmSymbols:
    def test_all_symbols_prefixed(self, libdrm_isolated, toolchain):
        _, result = libdrm_isolated
        vr = verify_symbols(result, "rocm_", SYMBOL_PATTERNS, toolchain)
        assert len(vr.prefixed_symbols) > 50
        assert not vr.unprefixed_leaks, f"Leaks: {vr.unprefixed_leaks[:10]}"


class TestLibdrmLinking:
    def test_autoconf_link(self, libdrm_isolated, toolchain, tmp_path):
        _, result = libdrm_isolated
        assert verify_autoconf_link(result, CONSUMER_FUNC, LINK_NAME, toolchain, tmp_path)

    def test_negative_link(self, libdrm_isolated, toolchain, tmp_path):
        _, result = libdrm_isolated
        assert verify_negative_link(result, CONSUMER_FUNC, toolchain, tmp_path)


class TestLibdrmRuntime:
    def test_isolated_only(self, libdrm_isolated, toolchain, tmp_path):
        _, result = libdrm_isolated
        ok, bindings = verify_runtime_isolation(
            result, CONSUMER_FUNC, LINK_NAME, "rocm_", toolchain, tmp_path,
        )
        assert ok, "\n".join(bindings[-5:])
