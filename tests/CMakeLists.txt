link_libraries(ssim)
find_package(GTest REQUIRED)
enable_testing()
macro(SSIM_ADD_EXECUTABLE directory)
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_LIST_DIR}/${directory}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/${directory}/*.cu)
  add_executable(${directory} ${sources})
endmacro()

macro(SSIM_ADD_TEST directory)
  file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/${directory}/*.cpp)
  add_executable(${directory} ${sources})
  target_link_libraries(${directory} GTest::GTest GTest::Main)
  add_test(NAME ${directory} COMMAND ${directory})
endmacro()

SSIM_ADD_TEST(fem_intro)
SSIM_ADD_TEST(fem_local_global)
SSIM_ADD_TEST(fem_time_step_diff)
SSIM_ADD_EXECUTABLE(fem_time_step)
SSIM_ADD_EXECUTABLE(fem_ext_ai)

if(SSIM_ENABLE_CUDA)
  SSIM_ADD_EXECUTABLE(fem_cu)
endif()
