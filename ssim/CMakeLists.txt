add_library(ssim INTERFACE)
target_include_directories(ssim INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(ssim INTERFACE mathprim::mathprim)

set(SSIM_CXX_COMPILE_OPTIONS "")
set(SSIM_CUDA_COMPILE_OPTIONS "")

if(SSIM_ENABLE_WARNINGS)
  if(MSVC)
    list(APPEND SSIM_CXX_COMPILE_OPTIONS /W4)
  else()
    list(APPEND SSIM_CXX_COMPILE_OPTIONS -Wall -Wextra -pedantic)
  endif()
endif()

target_compile_options(ssim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${SSIM_CXX_COMPILE_OPTIONS}>)
if(SSIM_ENABLE_CUDA)
  target_compile_options(ssim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${SSIM_CUDA_COMPILE_OPTIONS}>)
endif()
