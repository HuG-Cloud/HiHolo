cmake_minimum_required(VERSION 3.20)
project(holo_copying LANGUAGES C CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加调试信息编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# 查找必要的包
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
    ${GSL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../include
)

# 定义源文件
set(CPP_SRCS
    copy_holos.cpp
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
    ${MPI_CXX_LIBRARIES}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${HDF5_LIBRARIES}
    ${GSL_LIBRARIES}
)