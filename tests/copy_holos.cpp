#include <iostream>
#include <cstring>
#include "hdf5.h"

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <input file> <input dataset> <output file> <output dataset> <num angles>" << std::endl;
        return 1;
    }

    // 打开输入文件和数据集
    hid_t inFile = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t inDataset = H5Dopen2(inFile, argv[2], H5P_DEFAULT);

    // 获取输入数据的维度
    hid_t inSpace = H5Dget_space(inDataset);
    int ndims = H5Sget_simple_extent_ndims(inSpace);
    if (ndims != 3) {
        std::cout << "Input data must be 3-dimensional!" << std::endl;
        H5Sclose(inSpace);
        H5Dclose(inDataset);
        H5Fclose(inFile);
        return 1;
    }

    hsize_t dims[3];
    H5Sget_simple_extent_dims(inSpace, dims, NULL);
    
    // 读取输入数据
    float* inData = new float[dims[0] * dims[1] * dims[2]];
    H5Dread(inDataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, inData);

    // 创建输出文件
    hid_t outFile = H5Fcreate(argv[3], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // 设置输出数据的维度（增加角度维度）
    int numAngles = std::stoi(argv[5]); // 从命令行获取角度数量
    hsize_t outDims[4] = {static_cast<hsize_t>(numAngles), dims[0], dims[1], dims[2]};
    hid_t outSpace = H5Screate_simple(4, outDims, NULL);
    
    // 创建输出数据集
    hid_t outDataset = H5Dcreate2(outFile, argv[4], H5T_NATIVE_FLOAT, outSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // 分批次写入数据
    const int batchSize = 10;
    float* outData = new float[batchSize * dims[0] * dims[1] * dims[2]];
    for (int angle = 0; angle < numAngles; angle += batchSize) {
        int currentBatchSize = std::min(batchSize, numAngles - angle);
        std::cout << "正在处理角度 " << angle + 1 << " 到 " << angle + currentBatchSize << " / " << numAngles << std::endl;
        
        for (int i = 0; i < currentBatchSize; ++i) {
            std::memcpy(outData + i * dims[0] * dims[1] * dims[2], inData, dims[0] * dims[1] * dims[2] * sizeof(float));
        }
        
        // 定义数据集的超空间
        hsize_t offset[4] = {static_cast<hsize_t>(angle), 0, 0, 0};
        hsize_t count[4] = {static_cast<hsize_t>(currentBatchSize), dims[0], dims[1], dims[2]};
        hid_t memSpace = H5Screate_simple(4, count, NULL);
        H5Sselect_hyperslab(outSpace, H5S_SELECT_SET, offset, NULL, count, NULL);
        
        // 写入数据
        H5Dwrite(outDataset, H5T_NATIVE_FLOAT, memSpace, outSpace, H5P_DEFAULT, outData);
        H5Sclose(memSpace);
    }

    // 清理
    delete[] inData;
    delete[] outData;
    H5Sclose(inSpace);
    H5Dclose(inDataset);
    H5Fclose(inFile);
    H5Sclose(outSpace);
    H5Dclose(outDataset);
    H5Fclose(outFile);
    
    std::cout << "成功将3D数据扩展为4D数据！" << std::endl;
    std::cout << "输出维度: " << numAngles << " x " 
              << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;

    return 0;
}