cmake_minimum_required(VERSION 3.20)
project(test_calibrate LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加调试信息编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# 查找必要的包
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
    ${GSL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../include
)

# 定义源文件
set(CPP_SRCS
    test_calibrate.cpp
    ../src/math_utils.cpp
    ../src/image_utils.cpp
    ../src/io_utils.cpp
)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${CPP_SRCS})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${MPI_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${GSL_LIBRARIES}
)