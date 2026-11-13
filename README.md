# TinyGen

TinyGen converts TOSA MLIR models into standalone C/C++ inference code for microcontrollers.
The Python package invokes the native `tinyc` compiler, translates EmitC to C++, and collects
and specializes the required kernel sources.

## Install

Python 3.10+, CMake 3.22+, a C++17 compiler, and an installed **LLVM/MLIR 20.1.x**
toolchain are required to build the native compiler. LLVM must include MLIR's development
headers/libraries, CMake configuration, and `mlir-translate`. LLVM alone is insufficient.
The toolchain's shared libraries must also remain available when running the installed package;
wheels built here do not bundle the LLVM shared libraries.

```bash
git clone --recurse-submodules https://github.com/coslab-kr/tinygen.git
cd tinygen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install . -Ccmake.define.LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

The build installs `tinyc`, `mlir-translate`, and the kernel source assets inside the
`tinygen` package. Models can then be compiled from any working directory.
There is no import-time download or build. Use the compiler and linker matching your
LLVM distribution when its libraries contain LTO bitcode. For example, the official
macOS LLVM 20.1.2 archive requires its `clang`, `clang++`, and `ld64.lld`, rather
than an older system linker. Pass these through CMake's `CMAKE_C_COMPILER`,
`CMAKE_CXX_COMPILER`, and `CMAKE_EXE_LINKER_FLAGS` (`-fuse-ld=/path/to/ld64.lld`)
settings, including when building through pip.

For development, use `pip install -e .` with the same CMake setting.
To build a wheel, run `pip wheel .` with the same setting.

## Python API

```python
import tinygen

output = tinygen.compile_model("model.mlir", output_dir="gen")
print(output)  # absolute path to gen/model_tinygen/
```

The result contains the generated C++, EmitC MLIR, operator list, TinyGen adapters,
CMSIS-NN headers and required C sources, and the CMSIS-NN license.
The pipeline executes native tools in a temporary directory and only publishes a successful
result. An existing model output directory raises `FileExistsError`; use a new output
location for another run. Failed native commands raise `subprocess.CalledProcessError`.

To invoke only `tinyc` and save its EmitC output and operator list:

```python
emitc = tinygen.lower_model("model.mlir", output_dir="emitc")
```

To use separately built executables:

```python
output = tinygen.compile_model(
    "model.mlir",
    output_dir="gen",
    tinyc="/path/to/build/compiler/tinyc",
    mlir_translate="/path/to/llvm/bin/mlir-translate",
)
```

Tool lookup order is explicit argument, environment override (`TINYGEN_TINYC` or
`TINYGEN_MLIR_TRANSLATE`), bundled executable, source checkout's `build/compiler/`,
then `PATH`. An invalid explicit override raises an error. Use LLVM/MLIR 20.1.x
for both native tools. `optimize=False` disables kernel constant specialization.

## CLI

```bash
tinygen model.mlir -o gen
python -m tinygen model.mlir -o gen
tinygen model.mlir --emitc-only -o emitc
```

The CLI also accepts `--tinyc`, `--mlir-translate`, and `--no-optimize`.

## CMSIS-NN updates

The official https://github.com/ARM-software/CMSIS-NN repository is a Git submodule at
`third_party/CMSIS-NN`, initially pinned to **v8.0.0**. TinyGen-specific C++ wrappers
remain in `kernels/cmsis-nn`; upstream sources must not be edited in place.

After cloning an existing checkout:

```bash
git submodule update --init --recursive
```

To update to the newest stable version and verify compatibility before recording it:

```bash
git -C third_party/CMSIS-NN fetch --tags origin
# Select the newest stable vX.Y.Z tag from this list:
git -C third_party/CMSIS-NN tag --list 'v*' --sort=-version:refname
git -C third_party/CMSIS-NN checkout v8.0.0  # replace with the selected tag
PYTHONPATH=src python -m unittest discover -s tests -v
git add third_party/CMSIS-NN
# Commit the updated pointer together with any required adapter changes.
```

A submodule pins a reproducible commit; it does not update itself on installation.
The collector discovers transitive C-source dependencies from upstream function definitions
and references, including architecture-specific branches. The compatibility test compiles
and links every adapter with the collected sources on the host. It does not validate
Cortex-M DSP/MVE execution, numerical equivalence, or performance; those require target testing.

## Source layout

```text
src/tinygen/          Python API, CLI, kernel collection and optimization
compiler/
  main.cpp           native driver, explicit output paths and error handling
  mlir/lib/
    TosaToEmitCPass.cpp       module validation and lowering orchestration
    TosaToEmitC/Operators.cpp operator-specific rewrite patterns
    TosaToEmitC/TensorEmitter.cpp shared tensor initialization
    MemoryPlanner/           byte sizing and lifetime-aware arena allocation
  tests/             native planner and repeated-pass regression tests
kernels/             TinyGen adapters, reference kernels and tensor headers
third_party/CMSIS-NN/ upstream Git submodule
tests/               Python pipeline and CMSIS adapter compatibility checks
```

For a standalone native build:

```bash
cmake -S . -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build -j
```

For Python development with prebuilt native tools (without rebuilding `tinyc`):

```bash
python -m pip install -e . -Ccmake.define.TINYGEN_BUILD_COMPILER=OFF
PYTHONPATH=src python -m unittest discover -s tests -v
```

This development option does not include native executables; supply them through API
arguments, environment overrides, or `PATH`. Pipeline tests use fake native executables
to exercise Python orchestration independently of LLVM. Kernel tests use real host C/C++ compilers.

## Native validation

The current lowering accepts one defined, single-block `main` function with one
tensor return and nonempty static ranked tensor inputs/results. Nested control
flow and multi-result operations are rejected with diagnostics. The arena uses
byte-accurate element sizes, naturally aligned storage, distinct pinned inputs
and outputs, and reuse only after an intermediate's last use. Each compilation
owns its lowering state and def/use graph.

`tinyc` keeps its default output filenames for compatibility, and also supports:

```bash
build/compiler/tinyc model.mlir -o model_emitc.mlir --ops-output model.ops
```

Build native regression tests with the same LLVM/toolchain settings as the compiler:

```bash
cmake -S . -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DTINYGEN_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
TINYGEN_TEST_TINYC="$PWD/build/compiler/tinyc" \
TINYGEN_TEST_TRANSLATE=/path/to/llvm/bin/mlir-translate \
PYTHONPATH=src python -m unittest discover -s tests -v
```

Native tests check deterministic repeated passes, byte sizes and arena lifetimes,
then translate generated code, compile it with a host C++ compiler, and execute
float-add models including constants, broadcasting and branched dependencies.
They also check unsupported inputs and file-write failures. Without the two
`TINYGEN_TEST_*` variables, native Python tests are explicitly skipped.

## Supported operators

INT8: Convolution2D, DepthwiseConvolution2D, MaxPool2D, AveragePool2D,
TransposeConvolution2D, Transpose, Pad, Reshape, Add, Cast.

Float: Add, Sub, Multiplication, Maximum, Reciprocal, Exponential,
ReduceSum, ReduceMax, Cast.

## Paper

TinyGen: Portable and Compact Code Generation for Tiny Machine Learning

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

## Authors

Gaeun Ko and Seonyeong Heo
