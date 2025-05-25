cmake_minimum_required(VERSION 3.20)
project(holo_recons LANGUAGES CXX CUDA C)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加用户可配置的选项
option(USE_BUNDLED_DEPENDENCIES "使用捆绑的依赖项" OFF)

# 设置可配置的库路径
set(ARGPARSE_DIR "/home/hug/workspace/software/argparse" CACHE PATH "argparse 库的路径")
set(SimpleITK_DIR "/home/hug/workspace/software/SimpleITK-build/SimpleITK-build" CACHE PATH "SimpleITK 配置目录")
set(ITK_DIR "/home/hug/workspace/software/SimpleITK-build/ITK-build" CACHE PATH "ITK 配置目录")

# 查找必要的包
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED)
find_package(SimpleITK REQUIRED)

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    ${ARGPARSE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_PATH}
)

# 定义公共源文件
set(COMMON_CPP_SRCS
    src/math_utils.cpp
    src/imageio_utils.cpp
    src/ProjectionSolver.cpp
)

set(COMMON_CUDA_SRCS
    src/Propagator.cu
    src/WaveField.cu
    src/Projector.cu
    src/holo_recons.cu
    src/cuda_utils.cu
)

# 定义所有应用程序
set(APP_TARGETS
    holo_recons_interval
    holo_recons_ctf
    holo_recons_ctf_angles
    holo_recons_ite_angles
    holo_distance_calibr
    holo_data_preprocess
    holo_data_prepro_angles
)

# 为每个应用程序创建可执行文件
add_executable(holo_recons_interval examples/arg_recons_interval.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_recons_ctf examples/arg_recons_ctf.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_recons_ctf_angles examples/arg_recons_ctf_angles.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_recons_ite_angles examples/arg_recons_ite_angles.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_distance_calibr examples/arg_distance_calibr.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_data_preprocess examples/arg_data_prepro.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})
add_executable(holo_data_prepro_angles examples/arg_prepro_angles.cpp ${COMMON_CPP_SRCS} ${COMMON_CUDA_SRCS})

# 定义公共链接库
set(COMMON_LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${HDF5_LIBRARIES}
    ${SimpleITK_LIBRARIES}
    ${GSL_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
)

# 为每个应用程序链接库
foreach(target ${APP_TARGETS})
    target_link_libraries(${target} ${COMMON_LIBS})
    # 添加MPI编译定义
    target_compile_definitions(${target} PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
    # 添加MPI编译选项
    target_compile_options(${target} PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
    # 添加MPI链接选项
    target_link_options(${target} PRIVATE ${MPI_CXX_LINK_FLAGS})
endforeach()