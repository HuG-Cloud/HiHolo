#ifndef IMAGEIO_UTILS_H_
#define IMAGEIO_UTILS_H_

#include <hdf5.h>
#include <gsl/gsl_fit.h>
#include <opencv2/opencv.hpp>
#include <SimpleITK.h>
#include "datatypes.h"
#include "math_utils.h"

extern const uchar maxUChar;

namespace ImageUtils
{   
    // Construct a vector containing multidistance cv::Mats from a complete set of data
    std::vector<cv::Mat> convertVecToMats(const U16Array &data, int numImages, int rows, int cols);
    cv::Mat convertVecToMat(const U16Array &data, int rows, int cols);
    FArray convertMatsToVec(const std::vector<cv::Mat> &mats);
    void convertVecToImgs(float *data, std::vector<itk::simple::Image> &images, int rows, int cols);
    void convertMatsToImgs(const std::vector<cv::Mat> &mats, std::vector<itk::simple::Image> &images, int rows, int cols);
    void convertImgsToVec(const std::vector<itk::simple::Image> &images, float *data, int rows, int cols);
    
    void removeOutliers(cv::Mat &originalImg, int kernelSize = 3, float threshold = 2.0f);    
    cv::Mat genCorrMatrix(const cv::Mat &image, int range, int windowSize);
    // Remove stripes of different dimensions
    void removeStripes(cv::Mat &image, int rangeRows = 0, int rangeCols = 0,
                       int windowSize = 5, const std::string &method = "mul");

    IntArray registerImage(const itk::simple::Image &fixedImage, itk::simple::Image &movingImage);
    D2DArray calibrateDistance(const FArray &holograms, int numImages, int rows, int cols,
                               double length, double pixelSize, const DArray &nz, double stepSize);
    double computePSD(const cv::Mat &image);

    void displayNDArray(F2DArray &images, int rows, int cols, const std::vector<std::string> &imgName);
    void displayPhase(FArray &phase, int rows, int cols, const std::string &imgName);
}

namespace IOUtils
{   
    bool readRawData(const std::string &filename, const std::vector<std::string> &datasetNames,
                     std::vector<hsize_t> &dims, U16Array &data, U16Array &dark, U16Array &flat);
    bool readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims);
    bool readPhasegrams(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims);
    bool readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims);
    bool saveProcessedGrams(const std::string &filename, const std::string &datasetName, const FArray &processedGrams, int numImages, int rows, int cols);
    bool savePhasegrams(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols);
    bool save3DGrams(const std::string &filename, const std::string &datasetName, const FArray &registeredGrams, int numImages, int rows, int cols);
    bool read3DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count);
    bool read4DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count);
    bool createFileDataset(const std::string &filename, const std::string &datasetName, const std::vector<hsize_t> &dims);
    bool write3DimData(const std::string &filename, const std::string &datasetName, const FArray &data, const std::vector<hsize_t> &dims, hsize_t offset);
    bool write4DimData(const std::string &filename, const std::string &datasetName, const FArray &data, const std::vector<hsize_t> &dims, hsize_t offset);
    bool saveImage(const std::string &filename, const FArray &image, int rows, int cols);
}

#endif