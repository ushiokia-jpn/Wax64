#!/usr/bin/env bash
# WASIX build script for jart/blink (Minix64) using wasixcc + CheerpX-style Wasm-JIT (Fork/Exec, no X11)
# Usage: INITIAL_MEMORY_BYTES=67108864 MAX_MEMORY_BYTES=2147483648 ./build_blink_wasix.sh [blink_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${1:-blink}"
INITIAL_MEMORY_BYTES="${INITIAL_MEMORY_BYTES:-67108864}"   # 64 MiB initial
MAX_MEMORY_BYTES="${MAX_MEMORY_BYTES:-2147483648}"         # 2 GiB max

if ! command -v wasixcc >/dev/null 2>&1; then
  echo "Error: wasixcc not found in PATH." >&2
  exit 1
fi

for req in wasm_jit.c; do
  if [ ! -f "$SCRIPT_DIR/$req" ]; then
    echo "Error: $SCRIPT_DIR/$req not found. Place $req next to this script." >&2
    exit 1
  fi
done

cd "$WORKDIR"
cp "$SCRIPT_DIR/wasm_jit.c" blink/wasm_jit.c

# Create WASIX compatibility header for missing POSIX/BSD macros & stubs
cat <<'EOF' > blink/wasix_compat.h
#ifndef BLINK_WASIX_COMPAT_H_
#define BLINK_WASIX_COMPAT_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <errno.h>
#include <grp.h>
#include <setjmp.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#ifndef sigsetjmp
#define sigsetjmp(env, savemask) setjmp(env)
#endif
#ifndef siglongjmp
#define siglongjmp(env, val) longjmp((env), (val))
#endif
#include <fcntl.h>
#include <limits.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

pid_t fork(void);

#ifndef PIPE_BUF
#define PIPE_BUF 4096
#endif

#ifndef F_GETLK
#define F_GETLK 5
#endif
#ifndef F_SETLK
#define F_SETLK 6
#endif
#ifndef F_SETLKW
#define F_SETLKW 7
#endif
#ifndef F_RDLCK
#define F_RDLCK 0
#endif
#ifndef F_WRLCK
#define F_WRLCK 1
#endif
#ifndef F_UNLCK
#define F_UNLCK 2
#endif

#ifndef PRIO_PROCESS
#define PRIO_PROCESS 0
#endif
#ifndef PRIO_PGRP
#define PRIO_PGRP 1
#endif
#ifndef PRIO_USER
#define PRIO_USER 2
#endif

#ifndef SO_DEBUG
#define SO_DEBUG 1
#endif

static inline int blink_wasix_fchdir(int fd) {
  (void)fd;
  errno = ENOSYS;
  return -1;
}
#define fchdir(fd) blink_wasix_fchdir(fd)

static inline int blink_wasix_getpriority(int which, id_t who) {
  (void)which;
  (void)who;
  return 0;
}
#define getpriority(which, who) blink_wasix_getpriority((which), (who))

static inline int blink_wasix_setpriority(int which, id_t who, int prio) {
  (void)which;
  (void)who;
  (void)prio;
  errno = ENOSYS;
  return -1;
}
#define setpriority(which, who, prio) blink_wasix_setpriority((which), (who), (prio))

static inline int blink_wasix_pause(void) {
  errno = EINTR;
  return -1;
}
#define pause() blink_wasix_pause()

static inline int blink_wasix_getgroups(int size, gid_t list[]) {
  (void)size;
  (void)list;
  return 0;
}
#define getgroups(size, list) blink_wasix_getgroups((size), (list))

int flock(int, int);

#endif /* BLINK_WASIX_COMPAT_H_ */
EOF

# Patch blink sources for WASIX + Wasm-JIT
python3 - <<'PY'
from pathlib import Path
import re

sc = Path("blink/syscall.c")
src = sc.read_text()
if "#ifndef __wasi__\n  _Static_assert(S_IFIFO == 0010000" not in src:
    pattern = r"(  _Static_assert\(S_IFIFO == 0010000.*?  _Static_assert\(S_IFMT == 0170000, \"\"\);[^\n]*\n)"
    src, n = re.subn(
        pattern,
        r"#ifndef __wasi__\n\1#endif\n",
        src,
        count=1,
        flags=re.DOTALL,
    )
    if n == 1:
        print("Patched S_IF* _Static_asserts in blink/syscall.c")

if "WasmJitReset();" not in src:
    src = src.replace(
        "  return pid;\n}\n\nstatic int SysFork",
        "  if (pid == 0) {\n#ifdef BLINK_WASM_JIT\n    extern void WasmJitReset(void);\n    WasmJitReset();\n#endif\n  }\n  return pid;\n}\n\nstatic int SysFork",
        1,
    )
    print("Patched Fork() WasmJitReset in blink/syscall.c")


sc.write_text(src)

bc = Path("blink/blink.c")
bsrc = bc.read_text()
if "BLINK_WASIX_LINK_STUBS" not in bsrc:
    bsrc += '''
#ifdef __wasi__
#define BLINK_WASIX_LINK_STUBS 1
int mprotect(void *addr, size_t len, int prot) {
  (void)addr;
  (void)len;
  (void)prot;
  return 0;
}
int flock(int fd, int op) {
  (void)fd;
  (void)op;
  return 0;
}
#endif
'''
    bc.write_text(bsrc)
    print("Added mprotect/flock stubs to blink/blink.c")

mc = Path("blink/map.c")
msrc = mc.read_text()
old_m = "static int GetBitsInAddressSpace(void) {\n#ifdef __EMSCRIPTEN__"
new_m = "static int GetBitsInAddressSpace(void) {\n#if defined(__EMSCRIPTEN__) || defined(__wasi__)"
if old_m in msrc:
    mc.write_text(msrc.replace(old_m, new_m, 1))
    print("Patched GetBitsInAddressSpace in blink/map.c for __wasi__")

lc = Path("blink/loader.c")
lsrc = lc.read_text()
old_l = '''  if (!IsWindows() && !(st->st_mode & 0111) && !IsBinFile(prog)) {
    LOGF("execve needs chmod +x");
    errno = EACCES;
    return -1;
  }'''
new_l = '''#ifndef __wasi__
  if (!IsWindows() && !(st->st_mode & 0111) && !IsBinFile(prog)) {
    LOGF("execve needs chmod +x");
    errno = EACCES;
    return -1;
  }
#endif'''
if "#ifndef __wasi__\n  if (!IsWindows() && !(st->st_mode & 0111)" not in lsrc and old_l in lsrc:
    lc.write_text(lsrc.replace(old_l, new_l, 1))
    print("Patched CheckExecutableFile in blink/loader.c for __wasi__")

mach_c = Path("blink/machine.c")
mach_src = mach_c.read_text()
if '#include "blink/wasm_jit.c"' not in mach_src:
    mach_src = mach_src.replace(
        "#else\n  JitlessDispatch(DISPATCH_NOTHING);\n#endif",
        "#else\n#ifdef BLINK_WASM_JIT\n  if (WasmJitDispatch(m)) return;\n#endif\n  JitlessDispatch(DISPATCH_NOTHING);\n#endif",
        1,
    )
    mach_src = mach_src.replace(
        "void ExecuteInstruction(struct Machine *m) {",
        '#include "blink/wasm_jit.c"\n\nvoid ExecuteInstruction(struct Machine *m) {',
        1,
    )
    mach_c.write_text(mach_src)
    print("Patched blink/machine.c for WasmJitDispatch")
PY

rm -f config.h config.log
rm -rf o//

export WASIXCC_WASM_EXCEPTIONS=no
export CC=wasixcc
export CXX=wasix++
export LD=wasixld
export AR=wasixar
export NM=wasixnm
export RANLIB=wasixranlib
export CFLAGS="-include blink/wasix_compat.h -DBLINK_WASM_JIT=1"
export LDFLAGS="-Wl,--initial-memory=${INITIAL_MEMORY_BYTES} -Wl,--max-memory=${MAX_MEMORY_BYTES} -Wl,--export-table,--growable-table,--export=wax64_r8,--export=wax64_r16,--export=wax64_r32,--export=wax64_r64,--export=wax64_w8,--export=wax64_w16,--export=wax64_w32,--export=wax64_w64"

WASIXCC_RUN_WASM_OPT=no WASIXCC_AUTOCONF_WORKAROUNDS=yes ./configure --disable-jit --disable-ancillary

python3 - <<'PY2'
from pathlib import Path
cfg = Path("config.h")
csrc = cfg.read_text()
csrc = csrc.replace("// #define HAVE_FORK", "#define HAVE_FORK 1")
cfg.write_text(csrc)
print("Enabled HAVE_FORK in config.h")
PY2

JOBS="$(nproc 2>/dev/null || echo 4)"
WASIXCC_RUN_WASM_OPT=yes make -j"$JOBS" o//blink/blink

mkdir -p dist-wasix
cp o//blink/blink dist-wasix/blink.wasm
echo ""
echo "Minix64 WASIX Wasm-JIT build complete: $WORKDIR/dist-wasix/blink.wasm"
