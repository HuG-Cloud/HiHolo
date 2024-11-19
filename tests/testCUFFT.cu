#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <chrono>

// 辅助函数:打印复数数组
void printComplexArray(const char* name, cuFloatComplex* arr, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << "(" << arr[i].x << "," << arr[i].y << ") ";
    }
    std::cout << std::endl;
}

int main() {
    const int rows = 5;
    const int cols = 5;
    const int batchSize = 1;
    // 分配主机内存
    cuFloatComplex* h_data = new cuFloatComplex[rows * cols * batchSize];
    
    // 初始化数据
    for (int i = 0; i < rows * cols * batchSize; i++) {
        h_data[i].x = static_cast<float>(i);
        h_data[i].y = 1.0f;
    }
    
    // 分配设备内存
    cuFloatComplex* d_data;
    cudaMalloc(&d_data, rows * cols * batchSize * sizeof(cuFloatComplex));
    
    // 将数据复制到设备
    cudaMemcpy(d_data, h_data, rows * cols * batchSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    
    // 创建 CUFFTUtils 对象
    CUFFTUtils fftUtil(rows, cols, batchSize);
    
    // 测量GPU执行时间
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    
    // cudaEventRecord(start);
    
    // 执行正向 FFT
    fftUtil.fft_fwd_batch(d_data);
    
    // 执行反向 FFT
    fftUtil.fft_bwd_batch(d_data);
    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    
    // std::cout << "GPU执行时间: " << milliseconds << " 毫秒" << std::endl;
    
    // 将结果复制回主机
    cudaMemcpy(h_data, d_data, rows * cols * batchSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printComplexArray("原始数据", h_data,  rows * cols * batchSize);
    
    // 清理内存
    delete[] h_data;
    cudaFree(d_data);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    
    return 0;
}