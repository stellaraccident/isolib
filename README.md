# isolib — ELF Symbol Isolation for Bundled System Dependencies

Tool for renaming dynamic symbols in shared libraries to prevent interposition
when bundled copies coexist with system copies in the same process.

## What it does

Given `libzstd.so.1`, isolib produces four artifacts:

1. **Prefixed .so** — all exported symbols renamed (`ZSTD_decompress` → `rocm_ZSTD_decompress`)
   via direct ELF binary rewriting of `.dynsym`/`.dynstr` + hash table rebuild
2. **Trampoline .a** — asm stubs exporting original names, tail-jumping to prefixed names via PLT
3. **Linker script** — `INPUT(stubs.a AS_NEEDED(real.so.1))` installed as `-lzstd`
4. **Redirect header** — `#define` macros for compile-time bypass of trampolines

Autoconf `AC_CHECK_LIB(zstd, ZSTD_decompress)` works unmodified — the linker
script provides the original name via the trampoline, which pulls in the real .so.

## Usage

```bash
# Isolate a library
isolib isolate libzstd.so.1 --prefix rocm_ --name zstd -o output/

# Dry-run: see what would be renamed
isolib inspect libzstd.so.1

# Strict mode: treat warnings as errors
isolib isolate libzstd.so.1 --prefix rocm_ --name zstd -o output/ --werror

# Allow specific warning categories
isolib isolate libzstd.so.1 --prefix rocm_ --name zstd -o output/ --werror --allow-object-symbol
```

## Symbol classification

### What gets renamed

Only **defined, exported** symbols (GLOBAL or WEAK bind, DEFAULT or PROTECTED
visibility, not UND section) that don't match the CRT/glibc exclusion list.

### What does NOT get renamed

**Undefined imports** — symbols the library expects the linker to resolve from
other DSOs. These include:

- Standard libc imports (`malloc`, `printf`, `dlopen`, etc.)
- **Weak undefined hooks** — symbols like `ZSTD_trace_compress_begin` that
  libraries declare as optional callbacks. These are intentionally left as
  external binding points:
  - If nobody provides them, they resolve to NULL (weak semantics)
  - If a consumer provides them, they bind in as designed
  - Renaming them would break the hook mechanism entirely
  - The `-Bsymbolic` flag (applied by TheRock) ensures the library's own
    *defined* symbols still resolve internally — these weak imports are the
    only deliberate opening

**CRT/glibc symbols** — even if a library re-exports `malloc` (common for
allocator wrappers), we don't rename it. The CRT exclusion list covers ~150
common libc/pthread/math symbols plus glob patterns for `__cxa_*`, `__gcc_*`, etc.

### Warning categories

| Category | Trigger | Behavior |
|----------|---------|----------|
| `object-symbol` | OBJECT (global data) in exports | Renamed in .so, excluded from trampolines (can't trampoline data) |
| `tls-symbol` | TLS symbol in exports | Renamed in .so, excluded from trampolines |
| `ifunc-symbol` | GNU_IFUNC in exports | Trampolined via PLT (works but less tested) |
| `versioned-symbol` | Non-AMDROCM version tag | Renamed, version tag preserved |
| `weak-override` | WEAK symbol shadowing glibc name | Excluded from renaming |

Use `--werror` to make all warnings fatal, `--allow-<category>` to selectively
permit specific categories.

## ELF rewriting internals

`objcopy --redefine-syms` does **not** modify `.dynsym` (only `.symtab`), so
isolib includes its own ELF binary rewriter (`elf_rewrite.py`) that:

1. Builds a new `.dynstr` with renamed strings appended
2. Updates `.dynsym` `st_name` offsets
3. Rewrites `DT_SONAME` to match the output filename
4. Adds a new `PT_LOAD` segment for the grown `.dynstr`
5. Relocates the PHDR table if needed (borrowed from kpack patterns)
6. Rebuilds `.gnu.hash` and `.hash` tables with new name hashes
7. Updates `DT_STRTAB`/`DT_STRSZ` in `.dynamic`

## Testing

```bash
# Unit tests (synthetic test libraries)
pytest tests/

# Integration tests (real sysdep libraries from TheRock S3 mirror)
pytest integration/

# All tests
pytest tests/ integration/

# Dump before/after artifacts for manual inspection
pytest integration/ --dump-artifacts=/tmp/isolib-dump
```

Integration tests download real sysdep sources, build them, isolate, then verify:
symbol correctness (readelf), autoconf link simulation, negative link test, and
runtime isolation via `LD_DEBUG=bindings` scraping.
