cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/installed")
option(SSIM_BUILD_TESTS "Build tests" ON)
option(SSIM_ENABLE_WARNINGS "Enable warnings on test cases." ON)
option(SSIM_BUILD_BINDINGS "Build Python bindings" OFF)
option(SSIM_ENABLE_NATIVE_INSTR "Enable native instruction set" ON)

project(ssim)

find_package(mathprim)

if (NOT mathprim_FOUND)
  message(FATAL_ERROR "mathprim not found. Run get-mathprim.sh to install it to current folder")
endif()

# Inherit from mathprim's functionalities
set(SSIM_ENABLE_CUDA ${MATHPRIM_ENABLE_CUDA})

if (SSIM_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  enable_language(CUDA)
endif()

add_subdirectory(ssim)
if(SSIM_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(SSIM_BUILD_BINDINGS)
  add_subdirectory(bindings)
endif()