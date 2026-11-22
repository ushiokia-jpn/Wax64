# Wax64
"Wax64" is an experimental x86_64 Wasm emulator based on Jart/Blink and built on the @wasmer/sdk. It features an experimental "Hot-block Wasm JIT"—inspired by CheerpX—and supports `fork()` through the use of WASIX. It is capable of launching a Busybox environment via xterm_min.js. The project name is derived from Box64.
## How to use
Obtain `wasmer-sdk` and place it in the same directory as `Wax64.html` and `sw.js`. 
> (You might find the number of files to be quite small, but that is because many of the files used are statically linked within `Wax64.html`.)
Example:
```bash
npm install --no-save @wasmer/sdk@0.19.1
mkdir -p wasmer-sdk
cp -r node_modules/@wasmer/sdk/dist /path/to/wax64/dist
```
For the purpose of cross-origin-isolation, it is recommended to set up a simple server using a tool like Python. An example is provided below:
```python3
python3 -m http.server 8080 --directory /path/to/wax64/dist
```
