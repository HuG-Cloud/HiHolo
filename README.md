# FastHolo

## 项目概述

FastHolo 是一个专为X射线传播相位衬度成像（PBI）全息模式设计的高性能计算框架。面对现有软件在处理大规模、高分辨率数据时遇到的性能瓶颈与硬件支持限制，FastHolo 基于 C++/CUDA/MPI 混合并行架构，提供从数据预处理、相位恢复到三维层析重建的完整、高效解决方案。

本项目不仅实现了多种经典的相位恢复算法，更针对实际应用中的关键挑战，提出了三种创新的改进算法。这些优化使得 FastHolo 在保证重建质量的同时，性能远超同类主流软件，并展现了卓越的多GPU扩展能力，旨在为高能同步辐射光源（如HEPS）的前沿实验提供强大的数据处理支持。

## 项目结构

- `src/`: 包含项目的主要源代码
- `include/`: 包含项目的头文件
- `examples/`: 包含各种命令行应用程序示例
- `python/`: Python模块和绑定
- `tests/`: 包含项目的测试文件

## 主要功能

FastHolo 提供了一套完整的全息数据处理工具链，其核心功能围绕高性能相位恢复算法展开。

### 相位恢复算法

FastHolo 支持多种解析和迭代类型的相位恢复算法。

- **经典迭代算法**: 
  - AP (Alternating Projection)
  - RAAR (Relaxed Averaged Alternating Reflections)
  - HIO (Hybrid Input-Output)

- **核心改进算法**:
  - **AP with Probe (APWP)**: 借鉴ptychography思想，通过同步优化物体和探针波前，有效抑制由非理想光源或探针引入的伪影，提升重建保真度。
  - **Extrapolation Iteration (EPI)**: 采用计算外推技术，在不增加硬件成本的前提下，从有限视场的全息图中恢复高频信息，显著提升空间分辨率。
  - **Parallel IRP (PIRP)**: 针对计算密集型的三维迭代重建（IRP）算法，设计了基于MPI+CUDA的并行优化方案，将重建效率提升一个数量级以上。

- **解析算法**:
  - CTF (Contrast Transfer Function)

### 实验设置

- 距离标定 —— 源样本距离/源探测器距离

### 图像预处理

- 异常值和条纹去除
- 暗平场校正
- 全息图像配准

## 依赖

### 必需依赖

- **CUDA Toolkit** (>= 11.0)
- **OpenCV** (>= 4.0)
- **HDF5** (option: parallel)
- **GSL** (GNU Scientific Library)
- **MPI** (>=4.0, 用于多角度数据并行)
- **SimpleITK** (用于图像配准)

### 可选依赖

- **argparse** (命令行参数解析)

### Python模块额外依赖

- **Python** (>= 3.7)
- **pybind11**
- **numpy**
- **h5py**
- **matplotlib** (用于可视化)

## 安装与构建

### 1. 依赖安装

#### 系统级 Ubuntu/Debian

```bash
# 基础依赖
sudo apt update
sudo apt install cmake build-essential

# CUDA (请从NVIDIA官网下载对应版本)
# OpenCV
sudo apt install libopencv-dev

# HDF5/GSL
sudo apt install libhdf5-dev libhdf5-serial-dev libgsl-dev

# MPI
sudo apt install libopenmpi-dev openmpi-bin

# Python依赖
sudo apt install python3-dev python3-pip
pip3 install pybind11 numpy h5py matplotlib
```

#### Conda 环境

```bash
# 创建虚拟环境
conda create -n fastholo

# MPI/CUDA工具
conda install -c conda-forge openmpi cuda-cudart=12.6

# HDF5 with OpenMPI
conda install -c conda-forge "hdf5=1.14.6=mpi_openmpi*" openmpi

# OpenCV/GSL/CPP-Argparse
conda install -c conda-forge cpp-argparse libopencv gsl

# SimpleITK
conda install -c conda-forge libsimpleitk libitk-devel
```

### 2. C++/CUDA应用程序构建

#### 使用CMake

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Python模块构建

```bash
cd python
chmod +x build.sh
./build.sh
```

**或者手动构建：**

```bash
cd python
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 用户接口

### 1. 命令行应用程序

项目提供了多个专用的命令行工具，每个工具针对特定的重建任务：

#### 1.1 迭代重建 (`holo_recons_interval`)

```bash
./holo_recons_interval \
    --input_file input.h5 dataset \
    --output_file output.h5 phasedata \
    --fresnel_numbers 1e-4 2e-4 3e-4 \
    --iterations 200 \
    --algorithm 0 \
    --plot_interval 50 \
    --amplitude_limits 1 1 \
    --padding_size 50 50
```

**主要参数说明：**

- `-I, --input_file`: 输入HDF5文件和数据集名称
- `-O, --output_file`: 输出HDF5文件和数据集名称
- `-f, --fresnel_numbers`: 对应全息图的菲涅尔数列表
- `-i, --iterations`: 迭代次数
- `-a, --algorithm`: 算法选择 (0:AP, 1:RAAR, 2:HIO, 3:DRAP, 4:APWP, 5:BIPEPI)
- `-pi, --plot_interval`: 显示迭代间隔
- `-P, --algorithm_parameters`: 算法参数
- `-al, --amplitude_limits`: 振幅约束范围
- `-pl, --phase_limits`: 相位约束范围
- `-s, --support_size`: 支撑区域大小
- `-S, --padding_size`: 填充大小

#### 1.2 CTF重建 (`holo_recons_ctf`)

```bash
./holo_recons_ctf \
    --input_files input.h5 dataset \
    --output_files output.h5 phasedata \
    --fresnel_numbers 1e-4 2e-4 3e-4 \
    --ratio 0.1 \
    --low_freq_lim 1e-3 \
    --high_freq_lim 1e-1
```

**主要参数说明：**

- `-r, --ratio`: 吸收和相位偏移的固定比值
- `-L, --low_freq_lim`: 低频正则化参数
- `-H, --high_freq_lim`: 高频正则化参数

#### 1.3 多角度迭代重建 (`holo_recons_ite_angles`)

```bash
mpirun -n 4 ./holo_recons_ite_angles \
    --input_files input.h5 dataset \
    --output_files output.h5 phasedata \
    --batch_size 100 \
    --fresnel_numbers 1e-4 2e-4 3e-4 \
    --device_numbers 2
    --iterations 200 \
    --algorithm 0
```

**主要参数说明：**

- `-b, --batch_size`: 批处理大小
- `-d, --device_numbers`: 使用的GPU数量

#### 1.4 多角度CTF重建 (`holo_recons_ctf_angles`)

```bash
mpirun -n 4 ./holo_recons_ctf_angles \
    --input_files input.h5 dataset \
    --output_files output.h5 phasedata \
    --batch_size 10 \
    --fresnel_numbers 1e-4 2e-4 3e-4 \
    --device_numbers 2
```

#### 1.5 距离标定 (`holo_distance_calibr`)

```bash
./holo_distance_calibr \
    --input_file input.h5 dataset \
    --period_length 1e-6 \
    --pixel_size 6.5e-6 \
    --step_size 1e-3 \
    --num_steps 10 20 30
```

### 2. Python模块接口

Python模块 `fastholo` 提供了完整的API接口，支持脚本编程。

#### 2.1 基本导入和枚举

```python
import fastholo
import numpy as np
import h5py

# 算法枚举
algorithm = fastholo.Algorithm.AP  # 或 RAAR, HIO, DRAP, APWP, BIPEPI

# 投影类型
projection_type = fastholo.ProjectionType.Averaged  # 或 Sequential, Cyclic

# 传播核类型
kernel_type = fastholo.PropKernelType.Fourier  # 或 Chirp, ChirpLimited

# 填充类型
padding_type = fastholo.PaddingType.Replicate  # 或 Constant, Fadeout
```

#### 2.2 图像预处理

```python
# 去除异常值
cleaned_image = fastholo.removeOutliers(image, kernelSize=5, threshold=2.0)

# 去除条纹
destriped_image = fastholo.removeStripes(
    image, 
    rangeRows=0, 
    rangeCols=0, 
    windowSize=5, 
    method="mul"
)

# 距离标定
parameters = fastholo.calibrateDistance(
    holograms,        # 全息图数据
    numImages,        # 图像数量
    rows, cols,       # 图像尺寸
    periodLength,     # 周期长度
    pixelSize,        # 像素大小
    numSteps,         # 步数列表
    stepSize          # 步长
)
```

#### 2.3 CTF重建

```python
# 单次CTF重建
phase = fastholo.reconstruct_ctf(
    holograms,            # 全息图数据,2D/3D numpy array
    fresnelNumbers,       # 菲涅尔数 [[f1], [f2], ...]
    lowFreqLim=1e-3,      # 低频限制
    highFreqLim=1e-1,     # 高频限制
    betaDeltaRatio=0.0,   # β/δ比值
    padSize=[],           # 填充大小
    padType=fastholo.PaddingType.Replicate,
    padValue=0.0
)

# 批处理CTF重建
ctf_reconstructor = fastholo.CTFReconstructor(
    batchSize=5,
    numImages=3,
    imSize=[2048, 2048],
    fresnelNumbers=[[1e-4], [2e-4], [3e-4]],
    lowFreqLim=1e-3,
    highFreqLim=1e-1,
    ratio=0.1
)

# 处理批次数据
result = ctf_reconstructor.reconsBatch(hologram_batch)
```

#### 2.4 迭代重建

```python
# 单次迭代重建
result = fastholo.reconstruct_iter(
    holograms,                       # 全息图数据，2D/3D numpy array
    fresnelNumbers,                  # 菲涅尔数
    iterations=200,                  # 迭代次数
    initialPhase=np.array([]),       # 初始相位猜测
    algorithm=fastholo.Algorithm.AP, # 算法选择
    algoParameters=[0.7],            # 算法参数
    minPhase=-float('inf'),          # 相位约束
    maxPhase=float('inf'),
    minAmplitude=0.0,                # 振幅约束
    maxAmplitude=float('inf'),
    support=[],                      # 支撑约束
    outsideValue=0.0,
    padSize=[200, 200],              # 填充大小
    padType=fastholo.PaddingType.Replicate,
    padValue=0.0,
    projectionType=fastholo.ProjectionType.Averaged,
    kernelType=fastholo.PropKernelType.Fourier,
    holoProbes=np.array([]),         # 探针数据 (APWP算法)
    initProbePhase=np.array([]),     # 初始探针相位
    calcError=False                  # 是否计算误差
)

# 返回值: [phase, amplitude, probe_phase?, step_errors?, pm_errors?]
reconstructed_phase = result[0]
reconstructed_amplitude = result[1]
```

#### 2.5 EPI算法

```python
result = fastholo.reconstruct_epi(
    holograms,                      # 全息图数据，2D/3D numpy array
    fresnelNumbers,                 # 菲涅尔数
    iterations=200,                 # 迭代次数
    initialPhase=np.array([]),      # 初始相位
    initialAmplitude=np.array([]),  # 初始振幅
    minPhase=-float('inf'),         # 约束参数
    maxPhase=float('inf'),
    minAmplitude=0.0,
    maxAmplitude=float('inf'),
    support=[],
    outsideValue=0.0,
    padSize=[]
    projectionType=fastholo.ProjectionType.Averaged,
    kernelType=fastholo.PropKernelType.Fourier,
    calcError=False
)
```

#### 2.6 批处理迭代重建

```python
reconstructor = fastholo.Reconstructor(
    batchSize=5,                  # 批处理大小
    numImages=3,                  # 每批图像数量
    imSize=[2048, 2048],          # 图像尺寸
    fresnelNumbers=[[1e-4], [2e-4], [3e-4]],
    iterations=200,               # 迭代次数
    algorithm=fastholo.Algorithm.RAAR,
    algoParams=[0.75, 0.99, 20],  # RAAR参数
    minPhase=-3.14,               # 约束参数
    maxPhase=3.14,
    minAmplitude=0.0,
    maxAmplitude=2.0,
    support=[1024, 1024],         # 支撑区域
    outsideValue=0.0,
    padSize=[250, 250],           # 填充参数
    padType=fastholo.PaddingType.Replicate,
    padValue=0.0,
    projType=fastholo.ProjectionType.Averaged,
    kernelType=fastholo.PropKernelType.Fourier
)

# 处理批次数据
result = reconstructor.reconsBatch(hologram_batch, initial_phase_batch)
```

## 性能优化建议

### 1. GPU内存管理

- 对于多角度全息数据，使用批处理接口 (`CTFReconstructor`, `Reconstructor`)
- 合理设置批处理大小，避免GPU内存溢出
- 使用适当的填充大小以避免FFT影响

### 2. 参数调优

- **菲涅尔数**: 根据实际实验几何精确计算
- **填充大小**: 一般为图像尺寸的10-50%
- **迭代次数**: 监控收敛情况确定合适的次数
- **约束参数**: 根据先验知识设置合理的物理约束

## 贡献

欢迎对本项目进行贡献！请提交Pull Request或报告Issue。

## 许可证

本项目遵循MIT许可证。请参阅LICENSE文件了解更多详细信息。

## 联系方式

- 项目主页: https://code.ihep.ac.cn/jrhu/holotomo_cuda/-/tree/main?ref_type=heads
- 技术支持: jrhu@ihep.ac.cn