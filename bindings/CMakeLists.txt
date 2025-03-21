find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE PYSSIM_SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

if(SSIM_ENABLE_CUDA)
  file(GLOB_RECURSE PYMP_CU "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
  list(APPEND PYSSIM_SRC ${PYMP_CU})

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  enable_language(CUDA)
endif()

message(STATUS "Building Python module: ${PYSSIM_SRC}")

nanobind_add_module(libssim NB_STATIC ${PYSSIM_SRC})

target_link_libraries(libssim PUBLIC ssim)

# Install the module
install(TARGETS libssim LIBRARY DESTINATION .)