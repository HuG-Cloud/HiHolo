CXX = g++
NVCC = nvcc

CXXFLAGS = -g -std=c++17 -I/usr/local/include/opencv4 -I/usr/local/cuda/include \
		   -I/usr/include/hdf5/serial -I/home/hug/workspace/software/argparse/include \
		   -I../include

NVCCFLAGS = -I../include

LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui \
		  -L/usr/local/cuda/lib64 -lcudart -lcufft -lnppidei -lnppc\
          -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 \
		  

# define output file
TARGET = holo_recons_ctf

# define source file
CPP_SRCS = arg_recons_ctf.cpp ../src/math_utils.cpp ../src/imageio_utils.cpp ../src/ProjectionSolver.cpp
CUDA_SRCS = ../src/Propagator.cu ../src/WaveField.cu ../src/Projector.cu ../src/holo_recons.cu ../src/cuda_utils.cu

# 定义依赖文件
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
OBJS = $(CPP_OBJS) $(CUDA_OBJS)

# 默认目标
all: $(TARGET)

# 链接目标文件
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS)

# 定义编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
