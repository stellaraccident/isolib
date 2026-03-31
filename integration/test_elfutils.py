"""Integration tests for elfutils (libelf, libdw, libasm) symbol isolation.

elfutils is the most complex sysdep — it has upstream symbol versioning
(ELFUTILS_0.192 etc.) across three sub-libraries. This tests that our
ELF rewriter handles versioned symbols correctly.
"""

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

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/elfutils-0.192.tar.bz2"
TARBALL_HASH = "543188f5f2cfe5bc7955a878416c5f252edff9926754e5de0c6c57b132f21d9285c9b29e41281e93baad11d4ae7efbbf93580c114579c182103565fe99bd3909"
TARBALL_HASH_ALGO = "sha512"


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


def _find_so(install_dir: Path, name: str) -> Path | None:
    for libdir in ["lib", "lib64"]:
        candidate = install_dir / libdir / name
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def elfutils_built() -> dict[str, BuiltLibrary]:
    """Build elfutils and return BuiltLibrary for each sub-library."""
    install_dir = CACHE_DIR / "built" / "elfutils-0.192"

    libelf = _find_so(install_dir, "libelf.so")
    if libelf is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH, TARBALL_HASH_ALGO)

        # elfutils uses autotools with several dependencies we can skip
        env = {**os.environ}
        configure = source_dir / "configure"
        if not os.access(configure, os.X_OK):
            configure.chmod(0o755)

        try:
            subprocess.run(
                [
                    str(configure),
                    f"--prefix={install_dir}",
                    "--enable-shared",
                    "--disable-static",
                    "--disable-debuginfod",
                    "--disable-libdebuginfod",
                    "--disable-demangler",
                ],
                check=True, capture_output=True, text=True,
                cwd=source_dir, env=env,
            )
            subprocess.run(
                ["make", "-j", str(os.cpu_count() or 4)],
                check=True, capture_output=True, text=True,
                cwd=source_dir, env=env,
            )
            subprocess.run(
                ["make", "install"],
                check=True, capture_output=True, text=True,
                cwd=source_dir, env=env,
            )
        except subprocess.CalledProcessError as e:
            pytest.skip(f"elfutils build failed: {e.stderr[-300:] if e.stderr else str(e)}")

    result = {}
    for lib_name, so_name, patterns, func, link in [
        ("libelf", "libelf.so", ["elf_*", "gelf_*", "elf32_*", "elf64_*", "nlist"], "elf_version", "elf"),
        ("libdw", "libdw.so", ["dwarf_*", "dwfl_*", "dwelf_*"], "dwarf_begin", "dw"),
        ("libasm", "libasm.so", ["asm_*", "disasm_*"], "asm_begin", "asm"),
    ]:
        so = _find_so(install_dir, so_name)
        if so is None or not so.exists():
            continue
        result[lib_name] = BuiltLibrary(
            name=lib_name, so_path=so.resolve(), install_dir=install_dir,
            symbol_patterns=patterns, consumer_func=func, link_name=link,
        )

    if not result:
        pytest.skip("No elfutils libraries found after build")
    return result


@pytest.fixture(scope="module")
def libelf_isolated(elfutils_built: dict, toolchain: Toolchain):
    if "libelf" not in elfutils_built:
        pytest.skip("libelf not built")
    built = elfutils_built["libelf"]
    output_dir = CACHE_DIR / "isolated" / "elfutils-0.192" / "libelf"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IsolationConfig(
        input_so=built.so_path, prefix="rocm_", output_dir=output_dir,
        output_name="elf",
        allow_categories={
            WarningCategory.OBJECT_SYMBOL,
            WarningCategory.VERSIONED_SYMBOL,
        },
    )
    result = isolate_library(config, toolchain)
    return config, result, built


class TestLibelfSymbols:
    def test_all_symbols_prefixed(self, libelf_isolated, toolchain):
        _, result, built = libelf_isolated
        vr = verify_symbols(result, "rocm_", built.symbol_patterns, toolchain)
        assert len(vr.prefixed_symbols) > 50, (
            f"Expected 50+ elf symbols, got {len(vr.prefixed_symbols)}"
        )
        assert not vr.unprefixed_leaks, f"Leaks: {vr.unprefixed_leaks[:10]}"

    def test_version_tags_preserved(self, libelf_isolated, toolchain):
        """Verify that ELFUTILS_* version tags survive renaming."""
        _, result, _ = libelf_isolated
        proc = subprocess.run(
            [str(toolchain.readelf), "--dyn-syms", "-W", str(result.prefixed_so)],
            capture_output=True, text=True, check=True,
        )
        # At least some symbols should still have version tags
        versioned_lines = [
            l for l in proc.stdout.splitlines()
            if "rocm_" in l and "@" in l
        ]
        assert len(versioned_lines) > 0, (
            "No versioned symbols found in isolated libelf — version tags may have been lost"
        )


class TestLibelfLinking:
    def test_autoconf_link(self, libelf_isolated, toolchain, tmp_path):
        _, result, built = libelf_isolated
        assert verify_autoconf_link(
            result, built.consumer_func, built.link_name, toolchain, tmp_path,
        )

    def test_negative_link(self, libelf_isolated, toolchain, tmp_path):
        _, result, built = libelf_isolated
        assert verify_negative_link(result, built.consumer_func, toolchain, tmp_path)


class TestLibelfRuntime:
    def test_isolated_only(self, libelf_isolated, toolchain, tmp_path):
        _, result, built = libelf_isolated
        ok, bindings = verify_runtime_isolation(
            result, built.consumer_func, built.link_name, "rocm_",
            toolchain, tmp_path,
        )
        assert ok, "\n".join(bindings[-5:])

    def test_cohabitation(self, elfutils_built, libelf_isolated, toolchain, tmp_path):
        if "libelf" not in elfutils_built:
            pytest.skip("libelf not built")
        _, result, built = libelf_isolated
        ok, bindings = verify_runtime_isolation(
            result, built.consumer_func, built.link_name, "rocm_",
            toolchain, tmp_path, system_so=built.so_path,
        )
        assert ok, "\n".join(bindings[-5:])
