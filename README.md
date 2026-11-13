# TinyGen

TinyGen generates standalone C/C++ inference code from TOSA MLIR for microcontrollers.
It lowers models through EmitC and bundles the required TinyGen and CMSIS-NN kernels.

```text
TOSA MLIR → tinyc → EmitC → C/C++ + kernel sources
```

## Installation

Requires **Python 3.10+**, **CMake 3.22+**, a **C++17 compiler**, and **LLVM/MLIR 20.1.x**
with development headers, libraries, CMake configuration, and `mlir-translate`.

```bash
git clone --recurse-submodules https://github.com/coslab-kr/tinygen.git
cd tinygen
python -m pip install . -Ccmake.define.LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

The package includes `tinyc`, `mlir-translate`, and kernel sources.

<details>
<summary>Build configuration</summary>

- Use `pip install -e .` for development, with the same CMake settings.
- To use existing executables, install with `-Ccmake.define.TINYGEN_BUILD_COMPILER=OFF`.
- LLVM shared libraries, when required by the toolchain, are not bundled.
- Use a matching compiler and linker for LLVM libraries built with LTO. The official
  macOS LLVM 20.1.2 archive requires its `clang`, `clang++`, and `ld64.lld`.
  Set `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, and
  `CMAKE_EXE_LINKER_FLAGS=-fuse-ld=/path/to/ld64.lld` through pip's
  `-Ccmake.define.<NAME>=<VALUE>` options.

</details>

## Usage

### Python

```python
import tinygen

output = tinygen.compile_model("model.mlir", output_dir="gen")
# Returns the absolute path to gen/model_tinygen/.

# Run tinyc only, without C++ generation.
emitc = tinygen.lower_model("model.mlir", output_dir="emitc")
```

### Command line

```bash
tinygen model.mlir -o gen
tinygen model.mlir --emitc-only -o emitc
```

Use `tinygen --help` for executable overrides and optimization options.
In Python, pass `tinyc=...`, `mlir_translate=...`, or `optimize=False` to `compile_model`.
Executable paths can also be set with `TINYGEN_TINYC` and `TINYGEN_MLIR_TRANSLATE`.

Generated output includes C++, EmitC MLIR, an operator list, and the required kernel
sources and licenses. Existing model output directories are not overwritten.

## Model support

Inputs must contain one defined, single-block `main` function returning one tensor.
Tensor shapes must be static and nonempty. Nested control flow and multi-result
operations are not supported.

| Data type | Operators |
| --- | --- |
| INT8 | Conv2D, DepthwiseConv2D, TransposeConv2D, MaxPool2D, AveragePool2D, Transpose, Pad, Reshape, Add, Cast |
| Float | Add, Sub, Mul, Maximum, Reciprocal, Exp, ReduceSum, ReduceMax, Cast |

## Development

```text
src/tinygen/          Python API and CLI
compiler/            MLIR lowering, tensor initialization, and memory planning
kernels/             TinyGen adapters and reference kernels
third_party/CMSIS-NN/ Official CMSIS-NN submodule, pinned to v8.0.0
tests/               Python and native integration tests
```

Run Python and host-kernel tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

<details>
<summary>Build and test the native compiler</summary>

Use the same LLVM and compiler settings as for installation.

```bash
cmake -S . -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DTINYGEN_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

TINYGEN_TEST_TINYC="$PWD/build/compiler/tinyc" \
TINYGEN_TEST_TRANSLATE=/path/to/llvm/bin/mlir-translate \
PYTHONPATH=src python -m unittest discover -s tests -v
```

Native integration tests check generated C++ execution on the host. They are skipped
when the two `TINYGEN_TEST_*` variables are unset. Cortex-M execution and performance
require separate target testing.

</details>

<details>
<summary>Update CMSIS-NN</summary>

Keep TinyGen adapters in `kernels/cmsis-nn/` separate from the
[upstream sources](https://github.com/ARM-software/CMSIS-NN).
To update, select a stable release tag, test it, and record the new submodule commit:

```bash
git -C third_party/CMSIS-NN fetch --tags origin
git -C third_party/CMSIS-NN tag --list 'v*' --sort=-version:refname
git -C third_party/CMSIS-NN checkout vX.Y.Z
PYTHONPATH=src python -m unittest discover -s tests -v
git add third_party/CMSIS-NN
```

Replace `vX.Y.Z` with the selected release. Package installation uses the pinned
commit; it does not automatically fetch newer versions.

</details>

## Citation

Gaeun Ko and Seonyeong Heo. **TinyGen: Portable and Compact Code Generation for
Tiny Machine Learning.** In *Proceedings of the 35th ACM SIGPLAN International
Conference on Compiler Construction (CC ’26)*, pp. 131–142, 2026.
[DOI: 10.1145/3771775.3786278](https://doi.org/10.1145/3771775.3786278)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{10.1145/3771775.3786278,
author = {Ko, Gaeun and Heo, Seonyeong},
title = {TinyGen: Portable and Compact Code Generation for Tiny Machine Learning},
year = {2026},
isbn = {9798400722745},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3771775.3786278},
doi = {10.1145/3771775.3786278},
booktitle = {Proceedings of the 35th ACM SIGPLAN International Conference on Compiler Construction},
pages = {131–142},
numpages = {12},
location = {Sydney, NSW, Australia},
series = {CC '26}
}
```

</details>
