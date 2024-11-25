#include <iostream>
#include "../include/cuda_utils.h"

// 辅助函数:打印数组
void printArray(const char* name, float* arr, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int size = 4;
    
    // 分配主机内存
    FArray h_data {0.29, -0.11, 3.1, -2.9};
    
    // 分配设备内存
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    
    // 将数据复制到设备
    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 定义kernel配置
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 执行erfc计算
    compute_erfc<<<numBlocks, blockSize>>>(d_data, size);
    
    // 将结果复制回主机
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printArray("erfc计算结果", h_data.data(), size);
    
    // 清理内存
    cudaFree(d_data);
    
    return 0;
}
