# StreamBlocks MLIR

Experimental CAL support for MLIR

## Building

**Build and test LLVM/MLIR:**

```
$ cd circt
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON 
$ ninja
$ ninja check-
```
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-streamblocks
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

