"""Integration test infrastructure: download, build, cache real sysdep libraries.

Use --dump-artifacts=/path to copy before/after .so files for inspection:
    pytest integration/ --dump-artifacts=/tmp/isolib-dump
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dump-artifacts",
        default=None,
        help="Directory to dump before/after .so files for inspection",
    )


@pytest.fixture(scope="session")
def dump_dir(request: pytest.FixtureRequest) -> Path | None:
    """If --dump-artifacts is set, return the dump directory (created)."""
    val = request.config.getoption("--dump-artifacts")
    if val is None:
        return None
    p = Path(val)
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(autouse=True)
def _dump_isolation_artifacts(request: pytest.FixtureRequest, dump_dir: Path | None) -> None:
    """Auto-dump before/after artifacts when --dump-artifacts is set.

    For any test that uses a `*_built` and `*_isolated` fixture, copies:
      <dump_dir>/<lib>/before/<original.so>
      <dump_dir>/<lib>/after/<prefixed.so>
      <dump_dir>/<lib>/after/<stubs.a>
      <dump_dir>/<lib>/after/<linker_script>
      <dump_dir>/<lib>/after/<redirect.h>
    """
    if dump_dir is None:
        return

    # Look for *_built and *_isolated fixtures in the test's scope
    for name in list(request.fixturenames):
        if name.endswith("_built"):
            lib_name = name.removesuffix("_built")
            iso_name = f"{lib_name}_isolated"
            if iso_name not in request.fixturenames:
                continue

            try:
                built = request.getfixturevalue(name)
                _, result = request.getfixturevalue(iso_name)
            except Exception:
                continue

            lib_dir = dump_dir / lib_name
            before = lib_dir / "before"
            after = lib_dir / "after"
            before.mkdir(parents=True, exist_ok=True)
            after.mkdir(parents=True, exist_ok=True)

            # Copy original
            if hasattr(built, "so_path") and built.so_path.exists():
                shutil.copy2(built.so_path, before / built.so_path.name)

            # Copy isolated artifacts
            for attr in ("prefixed_so", "stubs_archive", "linker_script", "redirect_header"):
                p = getattr(result, attr, None)
                if p and p.exists():
                    shutil.copy2(p, after / p.name)
            break  # Only one library per test class

# Cache directory for downloaded sources and built artifacts
CACHE_DIR = Path(os.environ.get(
    "ISOLIB_CACHE_DIR",
    Path(__file__).parent.parent / ".cache" / "isolib",
))


@dataclass
class LibrarySource:
    """Downloaded and extracted source tree for a sysdep library."""

    name: str
    version: str
    source_dir: Path


@dataclass
class BuiltLibrary:
    """A built shared library ready for isolation testing."""

    name: str
    so_path: Path          # The .so file
    install_dir: Path      # Prefix where headers/libs are installed
    symbol_patterns: list[str]  # Glob patterns for symbols that must be prefixed
    consumer_func: str     # A function name for autoconf-style link tests
    link_name: str         # -l name (e.g. "zstd")


def download_and_extract(
    url: str,
    expected_hash: str,
    hash_algo: str = "sha256",
) -> Path:
    """Download a tarball and extract it, caching the result.

    Args:
        url: URL to download.
        expected_hash: Expected hash of the downloaded file.
        hash_algo: Hash algorithm (sha256, sha512).

    Returns:
        Path to the extracted source directory.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Derive cache paths from URL
    filename = url.rsplit("/", 1)[-1]
    archive_path = CACHE_DIR / "archives" / filename
    extract_dir = CACHE_DIR / "sources" / filename.split(".tar")[0].split(".zip")[0]

    if extract_dir.exists():
        return extract_dir

    # Download if not cached
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"Downloading {url}...")
        urlretrieve(url, archive_path)

    # Verify hash
    h = hashlib.new(hash_algo)
    h.update(archive_path.read_bytes())
    actual = h.hexdigest()
    if actual.lower() != expected_hash.lower():
        archive_path.unlink()
        raise RuntimeError(
            f"Hash mismatch for {filename}: expected {expected_hash}, got {actual}"
        )

    # Extract
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_extract = extract_dir.with_suffix(".extracting")
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir()

    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_extract)
    elif ".tar" in filename:
        with tarfile.open(archive_path) as tf:
            tf.extractall(tmp_extract, filter="data")

    # Most archives have a single top-level directory
    children = list(tmp_extract.iterdir())
    if len(children) == 1 and children[0].is_dir():
        children[0].rename(extract_dir)
        tmp_extract.rmdir()
    else:
        tmp_extract.rename(extract_dir)

    return extract_dir


def build_cmake_library(
    source_dir: Path,
    install_dir: Path,
    cmake_args: list[str] | None = None,
    targets: list[str] | None = None,
) -> None:
    """Build a CMake project and install it."""
    build_dir = source_dir / "_build"
    build_dir.mkdir(exist_ok=True)

    args = [
        "cmake",
        f"-S{source_dir}",
        f"-B{build_dir}",
        "-GNinja",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]
    if cmake_args:
        args.extend(cmake_args)

    subprocess.run(args, check=True, capture_output=True, text=True)

    build_cmd = ["ninja", "-C", str(build_dir)]
    if targets:
        build_cmd.extend(targets)
    subprocess.run(build_cmd, check=True, capture_output=True, text=True)

    subprocess.run(
        ["ninja", "-C", str(build_dir), "install"],
        check=True, capture_output=True, text=True,
    )


def build_autotools_library(
    source_dir: Path,
    install_dir: Path,
    configure_args: list[str] | None = None,
    make_args: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
) -> None:
    """Build an autotools project and install it."""
    env = {**os.environ}
    if env_extra:
        env.update(env_extra)

    configure = source_dir / "configure"
    # Ensure configure is executable (zip archives lose permissions)
    if configure.exists() and not os.access(configure, os.X_OK):
        configure.chmod(0o755)

    args = [str(configure), f"--prefix={install_dir}"]
    if configure_args:
        args.extend(configure_args)

    subprocess.run(args, check=True, capture_output=True, text=True, env=env,
                   cwd=source_dir)
    make_cmd = ["make", "-j", str(os.cpu_count() or 4)]
    if make_args:
        make_cmd.extend(make_args)
    subprocess.run(make_cmd, check=True, capture_output=True, text=True,
                   cwd=source_dir, env=env)
    subprocess.run(["make", "install"], check=True, capture_output=True, text=True,
                   cwd=source_dir, env=env)
