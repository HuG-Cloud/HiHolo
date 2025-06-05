cmake_minimum_required(VERSION 3.20)
project(fastholo LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找必要的包
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(CUDA REQUIRED)
find_package(GSL REQUIRED)

# 查找Python和pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../include
)

# 定义核心源文件
set(COMMON_CPP_SRCS
    ../src/math_utils.cpp
    ../src/image_utils.cpp
    ../src/ProjectionSolver.cpp
)

set(COMMON_CUDA_SRCS
    ../src/Propagator.cu
    ../src/WaveField.cu
    ../src/Projector.cu
    ../src/holo_recons.cu
    ../src/cuda_utils.cu
)

# 创建静态库
add_library(holo_recons_lib STATIC ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})

# 设置静态库的编译选项
set_target_properties(holo_recons_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(holo_recons_lib PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

# 为CUDA代码添加-fPIC选项
target_compile_options(holo_recons_lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

# 定义链接库
set(COMMON_LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${GSL_LIBRARIES}
)

target_link_libraries(holo_recons_lib ${COMMON_LIBS})
pybind11_add_module(fastholo bindings/pybind_wrapper.cpp)
target_link_libraries(fastholo PRIVATE holo_recons_lib)

# 编译器特定的定义
target_compile_definitions(fastholo PRIVATE VERSION_INFO=${PROJECT_VERSION})

# 设置输出属性
set_target_properties(fastholo PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)