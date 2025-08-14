#ifndef IMAGE_UTILS_H_
#define IMAGE_UTILS_H_

#include <opencv2/opencv.hpp>
#include <gsl/gsl_fit.h>
//#include <SimpleITK.h>
#include "math_utils.h"

extern const uchar maxUChar;

namespace ImageUtils
{   
    // Construct a vector containing multidistance cv::Mats from a complete set of data
    std::vector<cv::Mat> convertVecToMats(const U16Array &data, int numImages, int rows, int cols);
    std::vector<cv::Mat> convertVecToMats(const FArray &data, int numImages, int rows, int cols);
    cv::Mat convertVecToMat(const U16Array &data, int rows, int cols);
    cv::Mat convertVecToMat(const FArray &data, int rows, int cols);
    FArray convertMatsToVec(const std::vector<cv::Mat> &mats);
    FArray convertMatToVec(const cv::Mat &mat);
    //void convertVecToImgs(float *data, std::vector<itk::simple::Image> &images, int rows, int cols);
    //void convertMatsToImgs(const std::vector<cv::Mat> &mats, std::vector<itk::simple::Image> &images, int rows, int cols);
    //void convertImgsToVec(const std::vector<itk::simple::Image> &images, float *data, int rows, int cols);
    
    void removeOutliers(cv::Mat &originalImg, int kernelSize = 5, float threshold = 2.0f);
    cv::Mat genCorrMatrix(const cv::Mat &image, int range, int windowSize);
    // Remove stripes of different dimensions
    void removeStripes(cv::Mat &image, int rangeRows = 0, int rangeCols = 0,
                       int windowSize = 5, const std::string &method = "mul");

    //IntArray registerImage(const itk::simple::Image &fixedImage, itk::simple::Image &movingImage);
    //Int2DArray registerImages(float *data, int numImages, int rows, int cols);

    D2DArray calibrateDistance(const DArray &maxPSD, const DArray &nz, double length, double pixelSize, double stepSize);
    double computePSD(const cv::Mat &image, int direction, cv::Mat &profile, cv::Mat &fre);
    DArray computePSDs(const std::vector<cv::Mat> &images, int direction, std::vector<cv::Mat> &profiles, std::vector<cv::Mat> &frequencies);

    void displayNDArray(F2DArray &images, int rows, int cols, const std::vector<std::string> &imgName);
    void displayPhase(FArray &phase, int rows, int cols, const std::string &imgName);
    bool saveImage(const std::string &filename, const FArray &image, int rows, int cols);
}

#endif