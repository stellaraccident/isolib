"""Integration tests for elfutils (libelf, libdw, libasm) symbol isolation.

elfutils is the most complex sysdep:
- Upstream symbol versioning (ELFUTILS_0.192 etc.) across three sub-libraries
- Dependencies on other sysdeps (zstd, zlib, bzip2, liblzma) which must
  themselves be isolated first — elfutils must be built against the isolated
  versions so its imports reference rocm_ZSTD_* not ZSTD_*
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from isolib.model import IsolationConfig, IsolationResult, WarningCategory
from isolib.pipeline import isolate_library
from isolib.toolchain import Toolchain

from .conftest import BuiltLibrary, download_and_extract, CACHE_DIR
from .verify import verify_autoconf_link, verify_negative_link, verify_runtime_isolation, verify_symbols

TARBALL_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/elfutils-0.192.tar.bz2"
TARBALL_HASH = "543188f5f2cfe5bc7955a878416c5f252edff9926754e5de0c6c57b132f21d9285c9b29e41281e93baad11d4ae7efbbf93580c114579c182103565fe99bd3909"
TARBALL_HASH_ALGO = "sha512"

# Sysdep libraries that elfutils depends on — we must isolate these first
# and build elfutils against the isolated versions.
DEP_LIBS = {
    "zstd": {
        "url": "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zstd-1.5.7.tar.gz",
        "hash": "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3",
        "hash_algo": "sha256",
        "build": "cmake",
        "cmake_source_subdir": "build/cmake",
        "cmake_args": ["-DZSTD_BUILD_PROGRAMS=OFF", "-DZSTD_BUILD_TESTS=OFF", "-DZSTD_BUILD_STATIC=OFF"],
        "so_name": "libzstd.so",
        "iso_name": "zstd",
    },
    "zlib": {
        "url": "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zlib-1.3.2.tar.gz",
        "hash": "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
        "hash_algo": "sha256",
        "build": "cmake",
        "cmake_args": [],
        "so_name": "libz.so",
        "iso_name": "z",
    },
    "bzip2": {
        "url": "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/bzip2-1.0.8.tar.gz",
        "hash": "083f5e675d73f3233c7930ebe20425a533feedeaaa9d8cc86831312a6581cefbe6ed0d08d2fa89be81082f2a5abdabca8b3c080bf97218a1bd59dc118a30b9f3",
        "hash_algo": "sha512",
        "build": "makefile",
        "so_name": "libbz2.so",
        "iso_name": "bz2",
    },
    "liblzma": {
        "url": "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/xz-5.8.1.tar.bz2",
        "hash": "5965c692c4c8800cd4b33ce6d0f6ac9ac9d6ab227b17c512b6561bce4f08d47e",
        "hash_algo": "sha256",
        "build": "cmake",
        "cmake_args": ["-DBUILD_TESTING=OFF"],
        "so_name": "liblzma.so",
        "iso_name": "lzma",
    },
}


@pytest.fixture(scope="module")
def toolchain() -> Toolchain:
    return Toolchain.discover()


def _find_so(base: Path, name: str) -> Path | None:
    for d in ["lib", "lib64"]:
        c = base / d / name
        if c.exists():
            return c
    return None


def _build_and_isolate_dep(
    dep_name: str, dep_info: dict, tc: Toolchain,
) -> tuple[Path, Path]:
    """Build a dependency library, isolate it, return (isolated_dir, include_dir).

    The isolated_dir contains the linker script (e.g. libzstd.so) and the
    prefixed .so — this is what elfutils should link against.
    """
    from .conftest import build_cmake_library

    built_dir = CACHE_DIR / "built" / f"{dep_name}-elfutils-chain"
    iso_dir = CACHE_DIR / "isolated" / f"{dep_name}-elfutils-chain"

    # Check if already done
    iso_dir.mkdir(parents=True, exist_ok=True)
    linker_script = _find_so(iso_dir, dep_info["so_name"]) or (iso_dir / dep_info["so_name"])
    if linker_script.exists() and linker_script.stat().st_size > 0:
        include_dir = built_dir / "include"
        if not include_dir.exists():
            for d in ["include", "lib/include", "lib64/include"]:
                candidate = built_dir / d
                if candidate.exists():
                    include_dir = candidate
                    break
        return iso_dir, include_dir

    # Build
    source_dir = download_and_extract(
        dep_info["url"], dep_info["hash"], dep_info.get("hash_algo", "sha256"),
    )

    built_dir.mkdir(parents=True, exist_ok=True)

    if dep_info["build"] == "cmake":
        cmake_src = source_dir
        if "cmake_source_subdir" in dep_info:
            cmake_src = source_dir / dep_info["cmake_source_subdir"]
        build_cmake_library(cmake_src, built_dir, cmake_args=dep_info.get("cmake_args"))
    elif dep_info["build"] == "makefile":
        # bzip2 special case — Makefile build, manual install
        import shutil
        lib_dir = built_dir / "lib"
        lib_dir.mkdir(exist_ok=True)
        inc_dir = built_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["make", "-f", "Makefile-libbz2_so", "-j", str(os.cpu_count() or 4)],
            check=True, capture_output=True, text=True, cwd=source_dir,
        )
        for f in source_dir.glob("libbz2.so*"):
            shutil.copy2(f, lib_dir / f.name)
        so = lib_dir / "libbz2.so"
        if not so.exists():
            real = list(lib_dir.glob("libbz2.so.1.0*"))
            if real:
                so.symlink_to(real[0].name)
        # Install header
        for h in ["bzlib.h"]:
            src_h = source_dir / h
            if src_h.exists():
                shutil.copy2(src_h, inc_dir / h)

    so_path = _find_so(built_dir, dep_info["so_name"])
    if so_path is None:
        raise RuntimeError(f"Built {dep_name} but couldn't find {dep_info['so_name']} under {built_dir}")

    # Isolate
    config = IsolationConfig(
        input_so=so_path.resolve(),
        prefix="rocm_",
        output_dir=iso_dir,
        output_name=dep_info["iso_name"],
        allow_categories={WarningCategory.OBJECT_SYMBOL},
    )
    isolate_library(config, tc)

    # Find include dir
    include_dir = built_dir / "include"
    if not include_dir.exists():
        for d in [built_dir, built_dir / "lib", built_dir / "lib64"]:
            if (d / "include").exists():
                include_dir = d / "include"
                break

    return iso_dir, include_dir


@pytest.fixture(scope="module")
def isolated_deps(toolchain: Toolchain) -> dict[str, tuple[Path, Path]]:
    """Isolate all elfutils dependencies and return their paths."""
    deps = {}
    for name, info in DEP_LIBS.items():
        try:
            deps[name] = _build_and_isolate_dep(name, info, toolchain)
        except Exception as e:
            pytest.skip(f"Failed to build/isolate dependency {name}: {e}")
    return deps


@pytest.fixture(scope="module")
def elfutils_built(isolated_deps: dict, toolchain: Toolchain) -> dict[str, BuiltLibrary]:
    """Build elfutils against isolated dependencies."""
    install_dir = CACHE_DIR / "built" / "elfutils-0.192-chained"

    libelf = _find_so(install_dir, "libelf.so")
    if libelf is None:
        source_dir = download_and_extract(TARBALL_URL, TARBALL_HASH, TARBALL_HASH_ALGO)

        configure = source_dir / "configure"
        if not os.access(configure, os.X_OK):
            configure.chmod(0o755)

        # Build CPPFLAGS and LDFLAGS pointing at isolated deps
        cppflags_parts = []
        ldflags_parts = []
        for dep_name, (iso_dir, include_dir) in isolated_deps.items():
            if include_dir and include_dir.exists():
                cppflags_parts.append(f"-I{include_dir}")
            ldflags_parts.append(f"-L{iso_dir}")
            ldflags_parts.append(f"-Wl,-rpath,{iso_dir}")

        env = {
            **os.environ,
            "CPPFLAGS": " ".join(cppflags_parts),
            "LDFLAGS": " ".join(ldflags_parts),
        }

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
            pytest.skip(f"elfutils build failed: {e.stderr[-500:] if e.stderr else str(e)}")

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
    def test_all_elf_symbols_prefixed(self, libelf_isolated, toolchain):
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
        versioned_lines = [
            l for l in proc.stdout.splitlines()
            if "rocm_" in l and "@" in l
        ]
        assert len(versioned_lines) > 0, (
            "No versioned symbols found — version tags may have been lost"
        )

    def test_dep_imports_are_prefixed(self, libelf_isolated, toolchain):
        """Critical: libelf's imports of ZSTD_*/inflate/etc. must reference
        rocm_* names (via stubs), not the original unprefixed names."""
        _, result, _ = libelf_isolated
        proc = subprocess.run(
            [str(toolchain.readelf), "--dyn-syms", "-W", str(result.prefixed_so)],
            capture_output=True, text=True, check=True,
        )
        dep_imports = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            ndx = parts[6]
            name = parts[7].split("@")[0]
            if ndx != "UND":
                continue
            # Check for unprefixed dep symbols that should go through stubs
            for pattern in ["ZSTD_*", "inflate*", "deflate*", "BZ2_*", "lzma_*"]:
                import fnmatch
                if fnmatch.fnmatch(name, pattern) and not name.startswith("rocm_"):
                    dep_imports.append(name)

        assert not dep_imports, (
            f"libelf still has unprefixed dependency imports — it was not "
            f"built against isolated deps:\n  {dep_imports[:10]}\n"
            f"These should be rocm_ZSTD_*, rocm_inflate, etc."
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
