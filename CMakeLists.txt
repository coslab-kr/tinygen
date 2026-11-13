cmake_minimum_required(VERSION 3.22)
project(TinyGen LANGUAGES C CXX)

option(TINYGEN_BUILD_TESTS "Build native compiler regression tests" OFF)
if(TINYGEN_BUILD_TESTS)
  enable_testing()
endif()

option(TINYGEN_BUILD_COMPILER "Build and package the native tinyc compiler" ON)
if(TINYGEN_BUILD_COMPILER)
  add_subdirectory(compiler)
endif()

set(CMSIS_NN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CMSIS-NN")
if(NOT EXISTS "${CMSIS_NN_DIR}/Include/arm_nnfunctions.h")
  message(FATAL_ERROR "CMSIS-NN is missing. Run: git submodule update --init --recursive")
endif()
# These are target-MCU sources, not libraries to compile for the host Python wheel.
install(DIRECTORY kernels/ DESTINATION tinygen/data/kernels
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp")
install(DIRECTORY "${CMSIS_NN_DIR}/Include" "${CMSIS_NN_DIR}/Source"
        DESTINATION tinygen/data/CMSIS-NN FILES_MATCHING PATTERN "*.h" PATTERN "*.c")
install(FILES "${CMSIS_NN_DIR}/LICENSE" DESTINATION tinygen/data/CMSIS-NN)
