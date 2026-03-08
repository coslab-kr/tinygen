cmake_minimum_required(VERSION 3.15)
set(LLVM_VERSION "20.1.2")
set(LLVM_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm")

include("${CMAKE_CURRENT_LIST_DIR}/InstallLLVM.cmake") 

set(CMAKE_C_COMPILER "${CLANG_BIN}" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${CLANGXX_BIN}" CACHE PATH "C++ compiler" FORCE)

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

project(tinyc LANGUAGES C CXX)

include("${CMAKE_CURRENT_LIST_DIR}/SetLLVMEnvironment.cmake")

set(LLVM_INCLUDE_DIR ${LLVM_INSTALL_DIR}/${LLVM_TARGET_NAME}/include)
set(LLVM_LIB_DIR     ${LLVM_INSTALL_DIR}/${LLVM_TARGET_NAME}/lib)

include_directories(${LLVM_INCLUDE_DIR} mlir/include)
link_directories(${LLVM_LIB_DIR})

add_library(tinyc-mlir STATIC
  mlir/lib/TosaToEmitCPass.cpp
  mlir/lib/MemoryPlanner/LivenessAnalysis.cpp
  mlir/lib/MemoryPlanner/MemoryPlanner.cpp
  mlir/lib/MemoryPlanner/TensorGraph.cpp
  mlir/lib/MemoryPlanner/TensorNode.cpp
)

target_compile_options(tinyc-mlir PRIVATE -stdlib=libc++)

add_executable(tinyc main.cpp)

target_compile_options(tinyc PRIVATE -stdlib=libc++)
target_link_options(tinyc PRIVATE 
    -stdlib=libc++ 
    -fuse-ld=lld
    -static-libstdc++ 
    -static-libgcc
    -Wl,--allow-shlib-undefined
)

set_target_properties(tinyc PROPERTIES
  BUILD_RPATH "${LLVM_LIB_DIR}/x86_64-unknown-linux-gnu"
  INSTALL_RPATH "${LLVM_LIB_DIR}/x86_64-unknown-linux-gnu"
)

target_link_libraries(tinyc
  PRIVATE
    tinyc-mlir
    c++
    c++abi
    unwind
    pthread
    dl
    MLIRLoopLikeInterface 
    MLIRFunctionInterfaces
    MLIRTransformUtils
    MLIRRewrite
    MLIRRewritePDL
    MLIRPDLToPDLInterp
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRTosaDialect
    MLIRQuantDialect
    MLIRQuantUtils
    MLIRTensorDialect
    MLIRTensorUtils
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRMemRefDialect
    MLIRLLVMIRTransforms
    MLIRCallInterfaces
    MLIRMemRefUtils
    MLIRSparseTensorDialect
    MLIRComplexDialect
    MLIRArithDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRUBDialect
    MLIRControlFlowDialect
    MLIREmitCDialect
    MLIRDialect
    MLIRMemorySlotInterfaces
    MLIRDialectUtils
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRControlFlowInterfaces
    MLIRDestinationStyleOpInterface
    MLIRInferIntRangeInterface
    MLIRInferIntRangeCommon
    MLIRInferTypeOpInterface
    MLIRShapedOpInterfaces
    MLIRSideEffectInterfaces
    MLIRParallelCombiningOpInterface
    MLIRViewLikeInterface
    MLIRParser
    MLIRAsmParser
    MLIRPass
    MLIRBytecodeReader
    MLIRIR
    MLIRSupport
    LLVMSupport
    LLVMDemangle
  )
