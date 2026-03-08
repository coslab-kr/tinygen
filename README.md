# TinyGen

## Abstract

TinyGen is a code generation framework that produces standalone C/C++ inference code for running TinyML models on microcontrollers, without relying on any framework runtime.

TinyGen takes models in **TOSA MLIR** format as input and lowers them to the **EmitC dialect** before generating portable C/C++ source code.

- **LLVM Version**: 20.1.2
- **CMake**: 3.28.3

---

## Project Structure

```
tinygen-ae/
├── CMakeLists.txt
│
├── compiler/
│   ├── CMakeLists.txt
│   ├── InstallLLVM.cmake
│   ├── SetLLVMEnvironment.cmake
│   ├── main.cpp
│   └── mlir/
│       ├── include/
│       └── lib/
│           ├── MemoryPlanner/
│           └── TosaToEmitCPass.cpp
│
├── kernels/
│   ├── cmsis-nn/
│   ├── model.h
│   ├── reference/
│   └── tensor.h
│
├── scripts/
│   ├── copy_kernel.py
│   ├── define_ops.py
│   ├── main.py
│   └── params_prop.py
│
└── README.md
```

---

## Setup

### 1. Install CMake

```bash
sudo apt install cmake
```

### 2. Clone the Repository

```bash
git clone https://github.com/seonyheo/tinygen.git
cd tinygen
```

### 3. Build the Compiler

```bash
cmake -S . -B build
```
```bash
cmake --build build -j
```


### 4. Install Python and venv

```bash
sudo apt install python3
sudo apt install python3-venv
```

- Python Version: 3.12.3

### 5. Create a Virtual Environment

```bash
python3 -m venv {virtual environment's name}
```

### 6. Activate the Virtual Environment

```bash
source {virtual environment's name}/bin/activate
```

---

## Code Generation

### 1. Run the TinyGen Pipeline

From the project root:

```bash
python3 scripts/main.py model.mlir
```

This pipeline automatically:

- Converts TOSA IR to EmitC IR
- Generates standard C/C++ model source files
- Collects only the necessary files for linking
- Applies kernel-level optimizations (e.g., constant propagation)
- Gathers the generated sources and required runtime/kernel files into the gen/ directory for the final build/linking stage

---

## Supported Operators

| Operator | Type |
|----------|------|
| Convolution2D | INT8 |
| DepthwiseConvolution2D | INT8 |
| MaxPool2D | INT8 |
| AveragePool2D | INT8 |
| TransposeConvolution2D | INT8 |
| Transpose | INT8 |
| Pad | INT8 |
| Reshape | INT8 |
| Add | INT8, Float |
| Sub | Float |
| Multiplication | Float |
| Maximum | Float |
| Reciprocal | Float |
| Exponential | Float |
| ReduceSum | Float |
| ReduceMax | Float |
| Cast | INT8, Float |

---

## Paper

TinyGen: Portable and Compact Code Generation for Tiny Machine Learning

---

## Authors

- Gaeun Ko  
- Seonyeong Heo
