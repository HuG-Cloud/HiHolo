#include <chrono>
#include <iostream>
#include "image_utils.h"
#include "io_utils.h"

int main(int argc, char* argv[]) {
    const std::string inputPath = "/home/hug/Downloads/HoloTomo_Data/tomo_eps_data.hdf5";
    const std::string outputPath = "/home/hug/Downloads/HoloTomo_Data/eps_holo.h5";
   
    std::vector<hsize_t> dims;
    IOUtils::readDataDims(inputPath, "/scan/detector/data", dims);

    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    int numImages = 1;

    U16Array hologram, dark, flat1, flat2;
    IOUtils::read3DimData(inputPath, "/scan/detector/data", hologram, 1000, 1);
    IOUtils::read4DimData(inputPath, "/scan/detector/darks", dark, 0, 1);
    IOUtils::read4DimData(inputPath, "/scan/detector/flats", flat1, 0, 1);
    IOUtils::read4DimData(inputPath, "/scan/detector/flats", flat2, 1, 1);

    cv::Mat holoMat = ImageUtils::convertVecToMat(hologram, rows, cols);
    cv::Mat darkMat = ImageUtils::convertVecToMat(dark, rows, cols);
    cv::Mat flat1Mat = ImageUtils::convertVecToMat(flat1, rows, cols);
    cv::Mat flat2Mat = ImageUtils::convertVecToMat(flat2, rows, cols);

    std::cout << "原始图像尺寸: " << rows << "x" << cols << std::endl;

    ImageUtils::removeOutliers(holoMat);
    ImageUtils::removeOutliers(darkMat);
    ImageUtils::removeOutliers(flat1Mat);
    ImageUtils::removeOutliers(flat2Mat);

    cv::Mat flatMat = (flat1Mat + flat2Mat) / 2;
    holoMat = (holoMat - darkMat) / (flatMat - darkMat);
    
    // 裁剪图像 - 方法选择
    const int crop_border = 8;
    cv::Rect cropRect(crop_border / 2, crop_border, cols - crop_border, rows - crop_border);
    holoMat = holoMat(cropRect).clone();  // 使用clone()确保数据连续
    
    rows = holoMat.rows;
    cols = holoMat.cols;
    
    std::cout << "裁剪后图像尺寸: " << rows << "x" << cols << std::endl;
    
    //ImageUtils::removeStripes(holoMat);
    FArray holoVec = ImageUtils::convertMatToVec(holoMat);
    IOUtils::save3DGrams(outputPath, "holodata", holoVec, numImages, rows, cols);
    ImageUtils::saveImage("holo.png", holoVec, rows, cols);

    return 0;
}