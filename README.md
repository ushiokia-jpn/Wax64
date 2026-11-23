# Wax64
                                   
**Wax64** is an experimental x86_64 Wasm emulator based on [Jart/Blink](https://github.com/jart/blink) and built on `@wasmer/sdk`. It features an experimental "Hot-block Wasm JIT"—inspired by CheerpX—and supports `fork()` through the use of WASIX. It launches a Busybox environment via `xterm_min.js`. The project name is derived from Box64.

## Architecture

Wax64 runs x86_64 Linux binaries in the browser by combining an emulator core, a dynamic WebAssembly JIT, and WASIX process virtualization.

### 1. Tiered Wasm-JIT (`wasm_jit.c`)
* **Tier 1 & 2**: Uses `jart/blink` to profile hot code and compiles x86_64 blocks directly into raw WebAssembly bytecode without heavy dependencies like LLVM or Binaryen.
* **Dispatch & Caching**: Utilizes Wasm `br_table` for internal branching and caches all 16 GPRs and CPU flags in Wasm locals during execution.

### 2. Memory & Cache Optimization
* **Inline Software TLB**: Translates guest virtual addresses via separate read/write TLBs (`g_tlb_r` / `g_tlb_w`) to `SharedArrayBuffer` pointers using Wasm load/store instructions.
* **LRU Cache**: Employs a 4-way set-associative cache with automatic invalidation on memory calls (`mmap`, `munmap`, `mprotect`, `fork()`).

### 3. True `fork()` & Persistence
* **WASIX**: Built with `wasixcc` and `@wasmer/sdk` over `SharedArrayBuffer` and Web Workers to support real POSIX `fork()` and `execve()` in the browser.
* **OPFS Storage**: Connects the guest `/workspace` directory to the browser's Origin Private File System (OPFS) for persistence.

## How to use

Obtain `@wasmer/sdk` and place it in the same directory as `wax64.html` and `sw.js`.

> **Note:** You might find the number of files in this repository quite small—this is because many of the required assets are statically linked within `wax64.html`.

Example:

```bash
npm install --no-save @wasmer/sdk@0.19.1
mkdir -p wasmer-sdk
cp -r node_modules/@wasmer/sdk /path/to/blink/dist/wasmer-sdk
```

For cross-origin isolation (`SharedArrayBuffer` support), serve the directory using a local HTTP server such as Python:

```bash
python3 -m http.server 8080 --directory /path/to/blink/dist
```

## Demo image(Chromium, localhost)

<img width="1360" height="768" alt="Wax64 Demo" src="https://github.com/user-attachments/assets/e6f760ec-ed59-47a0-a3df-a59989cc8edc" />

> I designed the overall architecture, the Blink/WASIX integration, and debugged the edge cases, while heavily leveraging AI to accelerate writing the boilerplate Wasm bytecode emitter.
