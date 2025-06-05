# FastHolo
## 项目概述

该项目是一个基于C++/CUDA的全息重建系统，旨在对全息图数据进行预处理、相位恢复和CT重建，服务于Heps的Nano-holotomography实验。项目实现和优化了多种相位恢复算法，包括解析和迭代类型算法，并且性能优异。

## 项目结构

- `src/`: 包含项目的主要源代码。
- `include/`: 包含项目的头文件。
- `tests/`: 包含项目的测试文件。
- `examples/`: 包含项目的示例文件。

## 依赖

- CUDA Toolkit
- OpenCV
- HDF5
- Argparse

## 构建与运行

### 构建
确保已安装所有依赖项，然后在项目根目录下运行以下命令以构建项目：

```bash
make
```

### 运行

```bash
./holo_recons -I <hdf5_file> <dataset> -f <numbers> -i <num_iterations> ...
```
更多命令行参数请参考: `./holo_recons --help`

## 主要功能

- 全息图重建：使用CUDA加速的解析和迭代相位恢复算法进行二维重建。
- 全息图预处理：使用OpenCV、CUDA等进行全息图的数据预处理，包括去除噪声、图像配准等。
- CT重建：

## 贡献

欢迎对本项目进行贡献！请提交Pull Request或报告Issue。

## 许可证

本项目遵循MIT许可证。请参阅LICENSE文件了解更多详细信息。

