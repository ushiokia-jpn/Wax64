# Wax64
"Wax64", An experimental x86_64 Wasm emulator running on @wasmer/sdk, based on Jart/Blink. It features an experimental Hot-block-wasm-JIT inspired by CheerpX, and supports fork() thanks to WASIX. It boots into a Busybox environment via xterm_min.js. project name is inspired by Box64.
# How to use
For the purpose of cross-origin isolation, it is recommended to set up a simple server using a tool like Python. An example is provided below:
```python3
python3 -m http.server 8080 --directory /path/to/wax64/dist

```
