#include <iostream>
#include <chrono>

#include "../datatypes.h"
#include "../imageio_utils.h"

int main(void)
{   
    // cv::Mat mat1 = (cv::Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    cv::Mat mat1(00, 100, CV_32F);
    cv::randn(mat1, -100, 100);
    // cv::Mat mat2(100, 100, CV_32F);
    // cv::randn(mat2, -100, 100);
    // cv::Mat result = ImageUtils::xcorr2(mat1, mat2);
    // uniformMat.at<float>(2, 2) = 0;
    // uniformMat.at<float>(3, 1) = maxUInt_16;
    cv::Mat mat2;
    int dx = 2;
    int dy = 7;
    cv::Mat transMat = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(mat1, mat2, transMat, mat1.size());
    // cv::Mat mat2 = (cv::Mat_<float>(4,4) << 40,60,10,70,30,10,70,35,50,60,30,70,40,20,70,22);
    // std::cout << mat2 << std::endl << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // ImageUtils::removeOutliers(uniformMat);
    // std::cout << meanCols << std::endl;
    // cv::Mat mat2 = ImageUtils::genCorrMatrix(mat1, 1, 3);
    // std::cout << mat2 << std::endl;
    // ImageUtils::removeStripes(mat1);
    cv::Point2d shift = ImageUtils::alignImages01(mat2, mat1, true);
    // cv::Mat result;
    // cv::medianBlur(uniformMat, result, 3);
    // std::cout << mat1 -  alignedResult << std::endl << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << shift << std::endl;
    // std::cout << result << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time of situation 1: " << duration.count() << " million seconds" << std::endl;
    
}