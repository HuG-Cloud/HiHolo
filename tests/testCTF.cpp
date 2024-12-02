#include "../include/holo_recons.h"
#include "../include/imageio_utils.h"
#include <iostream>

int main() {
    try {
        // 创建一个更小的测试数据
        int rows = 10;
        int cols = 10;
        int numImages = 2;  // 使用2张图像
        
        // 创建测试用的全息图数据
        FArray holograms(rows * cols * numImages);
        for (int i = 0; i < numImages; i++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    // 创建一个简单的高斯形状
                    float x = (r - rows/2.0f) / (rows/4.0f);
                    float y = (c - cols/2.0f) / (cols/4.0f);
                    float value = exp(-(x*x + y*y));
                    holograms[i * rows * cols + r * cols + c] = value;
                }
            }
        }

        // 设置图像尺寸
        IntArray imSize = {rows, cols};
        
        // 设置Fresnel数
        F2DArray fresnelNumbers(numImages, FArray(2));
        fresnelNumbers[0] = {1.0f, 1.0f};
        fresnelNumbers[1] = {2.0f, 2.0f};

        // 设置CTF参数
        float lowFreqLim = 0.1f;
        float highFreqLim = 1.0f;
        float betaDeltaRatio = 0.1f;
        
        // 设置较小的padding
        IntArray padSize = {0, 0};  // 在每个维度上添加8个像素的padding
        
        // 打印输入数据
        std::cout << "输入全息图数据 (第一张图像):" << std::endl;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                std::cout << holograms[r * cols + c] << " ";
            }
            std::cout << std::endl;
        }

        // 执行CTF重建
        std::cout << "\n开始CTF重建..." << std::endl;
        FArray result = PhaseRetrieval::reconstruct_ctf(
            holograms, numImages, imSize, fresnelNumbers,
            lowFreqLim, highFreqLim, betaDeltaRatio,
            padSize, CUDAUtils::PaddingType::Replicate
        );

        // 打印完整的重建结果
        std::cout << "\n重建结果:" << std::endl;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                std::cout << result[r * cols + c] << " ";
            }
            std::cout << std::endl;
        }

        // F2DArray phase {result};
        // ImageUtils::displayNDArray(phase, rows, cols, std::vector<std::string>{"phase"});
        std::cout << "\nCTF重建测试完成!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
