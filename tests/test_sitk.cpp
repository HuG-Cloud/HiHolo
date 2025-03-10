#include <iostream>
#include <string>

#include <SimpleITK.h>

#include "../include/imageio_utils.h"

int main()
{
    // 1. 指定固定图像与移动图像 (使用TIF格式，并设定输出文件名)
    const std::string inputPath = "/home/hug/Downloads/HoloTomo_Data/holo_purephase.h5";
    const std::string outputPath1 = "/home/hug/Downloads/HoloTomo_Data/holo_shift_new.h5";
    const std::string outputPath2 = "/home/hug/Downloads/HoloTomo_Data/holo_regist_new.h5";
    const std::string datasetName = "holodata";
   
    std::vector<hsize_t> dims;
    IOUtils::readDataDims(inputPath, datasetName, dims);
    int numImages = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};
   
    FArray holograms;
    IOUtils::readProcessedGrams(inputPath, datasetName, holograms, dims);

    std::vector<itk::simple::Image> images(numImages);
    ImageUtils::convertVecToImgs(holograms.data(), images, rows, cols);

    std::vector<std::vector<double>> translations = {{8, 8}, {8, 8}, {8, 8}};
    // 计算最大填充边界
    std::vector<unsigned int> maxPadBound = {0, 0};
    for (int i = 1; i < numImages; i++) {
        maxPadBound[0] = std::max(maxPadBound[0], static_cast<unsigned int>(std::round(std::abs(translations[i-1][0]))));
        maxPadBound[1] = std::max(maxPadBound[1], static_cast<unsigned int>(std::round(std::abs(translations[i-1][1]))));
    }

    // 对所有图像进行填充
    for (int i = 1; i < numImages; i++) {
        images[i] = itk::simple::MirrorPad(images[i], maxPadBound, maxPadBound);
    }

    // 对填充后的图像进行变换和裁剪
    for (int i = 1; i < numImages; i++) {
        itk::simple::TranslationTransform transform(2);
        transform.SetParameters(translations[i-1]);
        images[i] = itk::simple::Resample(images[i], transform, itk::simple::sitkNearestNeighbor, 0.0, images[i].GetPixelID());
        images[i] = itk::simple::Extract(images[i], size, {static_cast<int>(maxPadBound[0]), static_cast<int>(maxPadBound[1])});
    }
    ImageUtils::convertImgsToVec(images, holograms.data(), rows, cols);
    IOUtils::save3DGrams(outputPath1, datasetName, holograms, numImages, rows, cols);
    std::cout << "Shifted image saved to: " << outputPath1 << std::endl;

    // 创建平移变换
    // std::vector<std::vector<double>> translations = {{5,0}, {0,8}, {6,8}};
    for (int i = 1; i < numImages; i++) {
        
        // 2. 设置平移变换
        // itk::simple::TranslationTransform transform(2);
        // transform.SetParameters(translations[i-1]);
        itk::simple::Transform transform = ImageUtils::registerImage(images[0], images[i]);
        std::cout << transform.GetParameters()[0] << " " << transform.GetParameters()[1] << std::endl;
        // 3. 对填充后的图像进行重采样
        // paddedImage = itk::simple::Resample(paddedImage, transform, itk::simple::sitkNearestNeighbor, 0.0, paddedImage.GetPixelID());        
    }

    ImageUtils::convertImgsToVec(images, holograms.data(), rows, cols);
    IOUtils::save3DGrams(outputPath2, datasetName, holograms, numImages, rows, cols);
    std::cout << "Registered image saved to: " << outputPath2 << std::endl;

    return 0;
}