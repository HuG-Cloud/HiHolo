CXX = g++
NVCC = nvcc

CXXFLAGS = -g -std=c++17 -I/usr/local/cuda/include -I../include

NVCCFLAGS = -I../include

LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcufft -lnppidei -lnppc
		  

# define output file
TARGET = testErfc

# define source file
CPP_SRCS = ../src/math_utils.cpp
CUDA_SRCS = ../src/cuda_utils.cu testErfc.cu

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
