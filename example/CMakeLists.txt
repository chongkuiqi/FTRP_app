cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# set(Torch_DIR /home/ckq/software/libtorch/share/cmake/Torch)
set(Torch_DIR /home/ckq/software/libtorch1.7.1/share/cmake/Torch)
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


set(OpenCV_DIR  /home/ckq/software/opencv-4.5.2/build)
find_package(OpenCV REQUIRED)

# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
# include_directories( /home/ckq/software/libtorch/include/torch/csrc/api/include/ )
include_directories( /usr/include/python3.8 )

# include_directories(-I/home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include -I/home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/TH -I/home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/THC -I/usr/local/cuda/include -I/home/ckq/miniconda3/envs/ckq/include/python3.8
# )
# include_directories( /home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/ )
# include_directories( /home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/ )


# add_executable(example-app example-app-MS.cpp)
# add_executable(example-app example-app.cpp)
# add_executable(example-app example-app.cpp roi_align_rotated_cuda.h roi_align_rotated_cuda.cpp roi_align_rotated_kernel.cu)
add_executable(example-app example-app.cpp roi_align_rotated_cpu.h roi_align_rotated_cpu.cpp )


target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})

message("打印")
message(${TORCH_LIBRARIES})

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)

