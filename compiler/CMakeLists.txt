# Use one installed LLVM/MLIR toolchain for headers, libraries and translation.
# Pass -DLLVM_DIR=/path/to/lib/cmake/llvm if it is not on CMake's search path.
find_package(LLVM 20.1 CONFIG REQUIRED)
if(NOT LLVM_VERSION_MAJOR EQUAL 20)
  message(FATAL_ERROR "tinyc requires LLVM/MLIR 20.1.x; found ${LLVM_PACKAGE_VERSION}")
endif()
find_package(MLIR CONFIG REQUIRED HINTS "${LLVM_DIR}/../mlir")

add_library(tinyc-mlir STATIC
  mlir/lib/TosaToEmitCPass.cpp
  mlir/lib/TosaToEmitC/Operators.cpp
  mlir/lib/TosaToEmitC/TensorEmitter.cpp
  mlir/lib/MemoryPlanner/LivenessAnalysis.cpp
  mlir/lib/MemoryPlanner/MemoryPlanner.cpp
  mlir/lib/MemoryPlanner/TensorGraph.cpp
  mlir/lib/MemoryPlanner/TensorNode.cpp
)
target_compile_features(tinyc-mlir PUBLIC cxx_std_17)
target_include_directories(tinyc-mlir SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_include_directories(tinyc-mlir PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/mlir/include")
separate_arguments(TINYGEN_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(tinyc-mlir PUBLIC ${TINYGEN_LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(tinyc-mlir PUBLIC /GR-)
  else()
    target_compile_options(tinyc-mlir PUBLIC -fno-rtti)
  endif()
endif()

target_link_libraries(tinyc-mlir PUBLIC
  MLIRParser MLIRPass MLIRTransforms MLIRTosaDialect MLIRQuantDialect
  MLIRQuantUtils MLIRTensorDialect MLIRTensorUtils MLIRArithDialect
  MLIRFuncDialect MLIRMemRefDialect MLIREmitCDialect
)
add_executable(tinyc main.cpp)
target_link_libraries(tinyc PRIVATE tinyc-mlir)
set_target_properties(tinyc PROPERTIES INSTALL_RPATH "${LLVM_LIBRARY_DIRS}")
install(TARGETS tinyc RUNTIME DESTINATION tinygen/bin)

find_program(TINYGEN_MLIR_TRANSLATE NAMES mlir-translate
  PATHS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
install(PROGRAMS "${TINYGEN_MLIR_TRANSLATE}" DESTINATION tinygen/bin)

if(TINYGEN_BUILD_TESTS)
  add_executable(tinygen-compiler-tests tests/CompilerTest.cpp)
  target_link_libraries(tinygen-compiler-tests PRIVATE tinyc-mlir)
  add_test(NAME tinygen-compiler COMMAND tinygen-compiler-tests)
endif()
