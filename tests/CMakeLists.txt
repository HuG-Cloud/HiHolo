cmake_minimum_required(VERSION 3.20)
project(distance_calibr_test LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(CUDA REQUIRED)
find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# SimpleITK 和 ITK 配置
set(SimpleITK_DIR "/home/hug/workspace/software/SimpleITK-build/SimpleITK-build" CACHE PATH "SimpleITK 配置目录")
set(ITK_DIR "/home/hug/workspace/software/SimpleITK-build/ITK-build" CACHE PATH "ITK 配置目录")
find_package(SimpleITK REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    /home/hug/workspace/software/argparse/include
    ../include
)

# 定义源文件
set(CPP_SRCS
    test_calibrate.cpp
    ../src/math_utils.cpp
    ../src/imageio_utils.cpp
    ../src/ProjectionSolver.cpp
)

set(CUDA_SRCS
    ../src/Propagator.cu
    ../src/WaveField.cu
    ../src/Projector.cu
    ../src/holo_recons.cu
    ../src/cuda_utils.cu
)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${CPP_SRCS} ${CUDA_SRCS})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${SimpleITK_LIBRARIES}
    ${GSL_LIBRARIES}
)