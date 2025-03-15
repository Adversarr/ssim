cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/installed")
option(SSIM_BUILD_TESTS "Build tests" ON)
option(SSIM_ENABLE_WARNINGS "Enable warnings on test cases." ON)

project(ssim)

find_package(mathprim)

if (NOT mathprim_FOUND)
  message(FATAL_ERROR "mathprim not found. Run get-mathprim.sh to install it to current folder")
endif()

# Inherit from mathprim's functionalities
set(SSIM_ENABLE_CUDA ${MATHPRIM_ENABLE_CUDA})

add_subdirectory(ssim)
if(SSIM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
