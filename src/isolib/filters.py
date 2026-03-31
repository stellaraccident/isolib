"""Symbol filtering: determine which symbols to prefix and which to exclude."""

from __future__ import annotations

import fnmatch
import re

from isolib.model import (
    ElfSymbol,
    IsolationWarning,
    SymbolBind,
    SymbolType,
    WarningCategory,
)

# CRT/compiler symbols that must never be renamed.
_CRT_PATTERNS: list[str] = [
    # C runtime
    "_init",
    "_fini",
    "_edata",
    "_end",
    "__bss_start",
    "__data_start",
    "data_start",
    # GCC/LLVM runtime
    "__cxa_*",
    "__gcc_*",
    "__gmon_*",
    "__stack_chk_*",
    "__asan_*",
    "__tsan_*",
    "__ubsan_*",
    "__msan_*",
    "__sanitizer_*",
    # Exception handling
    "__gxx_personality_v0",
    "_Unwind_*",
    # Glibc/POSIX - common symbols that libraries may reference
    "atexit",
    "malloc",
    "free",
    "calloc",
    "realloc",
    "posix_memalign",
    "aligned_alloc",
    "memalign",
    "valloc",
    "pvalloc",
    "abort",
    "exit",
    "_exit",
    "printf",
    "fprintf",
    "sprintf",
    "snprintf",
    "vprintf",
    "vfprintf",
    "vsprintf",
    "vsnprintf",
    "puts",
    "fputs",
    "fputc",
    "putc",
    "putchar",
    "fwrite",
    "fread",
    "fopen",
    "fclose",
    "fflush",
    "fseek",
    "ftell",
    "rewind",
    "feof",
    "ferror",
    "clearerr",
    "fileno",
    "open",
    "close",
    "read",
    "write",
    "lseek",
    "mmap",
    "munmap",
    "mprotect",
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "strlen",
    "strcmp",
    "strncmp",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "strchr",
    "strrchr",
    "strstr",
    "strerror",
    "strtol",
    "strtoul",
    "strtoll",
    "strtoull",
    "strtod",
    "strtof",
    "atoi",
    "atol",
    "atof",
    "qsort",
    "bsearch",
    "getenv",
    "setenv",
    "unsetenv",
    "dlopen",
    "dlclose",
    "dlsym",
    "dlerror",
    "pthread_*",
    "sem_*",
    "sched_*",
    # Math
    "sin",
    "cos",
    "tan",
    "sqrt",
    "pow",
    "log",
    "exp",
    "ceil",
    "floor",
    "fabs",
    "fmod",
    # errno
    "__errno_location",
    # Signal
    "signal",
    "sigaction",
    "raise",
    # System
    "sysconf",
    "getpagesize",
    "getpid",
    "getuid",
    "geteuid",
]

# Compile patterns once.
_CRT_RE: list[re.Pattern[str]] = [
    re.compile(fnmatch.translate(p)) for p in _CRT_PATTERNS
]


def _matches_crt(name: str) -> bool:
    """Check if a symbol name matches any CRT/glibc pattern."""
    return any(r.match(name) for r in _CRT_RE)


def _matches_patterns(name: str, patterns: list[str]) -> bool:
    """Check if name matches any of the given glob patterns."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def classify_symbol(
    sym: ElfSymbol,
    extra_exclude: list[str] | None = None,
) -> tuple[bool, IsolationWarning | None]:
    """Decide whether a symbol should be prefixed.

    Only DEFINED, GLOBAL/WEAK, DEFAULT/PROTECTED symbols are candidates
    for renaming. This means:

    - Undefined (UND) symbols are never renamed. These are imports that
      the library expects the runtime linker to resolve from other DSOs.
      This includes:

      * Standard libc imports (malloc, printf, etc.)
      * Weak undefined "hook" symbols (e.g. ZSTD_trace_compress_begin)
        that libraries use as optional callbacks. These are deliberately
        left unresolved — if a consumer provides them, they bind in.
        Renaming these would break the hook mechanism. The existing
        -Bsymbolic flag (applied by TheRock) ensures that defined
        symbols within the library still resolve internally, so these
        weak imports are the only deliberate external binding point.

    - CRT/glibc symbols (malloc, pthread_*, etc.) are excluded even if
      defined, to avoid breaking fundamental runtime linkage.

    Returns:
        (should_prefix, optional_warning)
        - should_prefix: True if the symbol should be renamed
        - warning: non-None if the symbol triggers a diagnostic
    """
    # Only prefix defined, exported symbols.
    # Undefined symbols (imports) are never renamed — see docstring.
    if not sym.is_exportable:
        return False, None

    # Never rename CRT/glibc symbols.
    if _matches_crt(sym.name):
        if sym.bind == SymbolBind.WEAK:
            return False, IsolationWarning(
                category=WarningCategory.WEAK_OVERRIDE,
                symbol_name=sym.name,
                message=f"WEAK symbol '{sym.name}' shadows a glibc name; excluded from renaming",
            )
        return False, None

    # User-specified exclusions.
    if extra_exclude and _matches_patterns(sym.name, extra_exclude):
        return False, None

    # TLS symbols: rename in .so but warn (no trampoline possible).
    if sym.is_tls:
        return True, IsolationWarning(
            category=WarningCategory.TLS_SYMBOL,
            symbol_name=sym.name,
            message=f"TLS symbol '{sym.name}' will be renamed but cannot be trampolined",
        )

    # OBJECT symbols: rename but warn (no trampoline).
    if sym.is_object:
        return True, IsolationWarning(
            category=WarningCategory.OBJECT_SYMBOL,
            symbol_name=sym.name,
            message=f"OBJECT symbol '{sym.name}' will be renamed but cannot be trampolined; use redirect header",
        )

    # IFUNC symbols: trampoline works via PLT but warn.
    if sym.sym_type == SymbolType.IFUNC:
        return True, IsolationWarning(
            category=WarningCategory.IFUNC_SYMBOL,
            symbol_name=sym.name,
            message=f"GNU_IFUNC symbol '{sym.name}' will be trampolined via PLT; less tested",
        )

    # Versioned symbols from upstream (not our AMDROCM tag).
    if sym.version and not sym.version.startswith("AMDROCM_"):
        return True, IsolationWarning(
            category=WarningCategory.VERSIONED_SYMBOL,
            symbol_name=sym.name,
            message=f"Symbol '{sym.name}' has upstream version tag '{sym.version}'",
        )

    return True, None
