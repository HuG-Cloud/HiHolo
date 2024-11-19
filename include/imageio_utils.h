#ifndef IMAGEIO_UTILS_H_
#define IMAGEIO_UTILS_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include "H5Cpp.h"
#include "datatypes.h"
#include "math_utils.h"

extern const uchar maxUChar;

namespace ImageUtils
{   
    enum Type {Constant, Replicate, Fadeout};

    template<class T>
    std::vector<T> cropMatrix(const std::vector<T> &matrix, int rows, int cols, int cropPreRows, int cropPreCols, int cropPostRows, int cropPostCols);

    template<class T>
    std::vector<T> padByConstant(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols, T padValue);

    template<class T>
    std::vector<T> padByReplicate(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols);

    // Pad matrix to given size in different ways
    template<class T>
    std::vector<T> padMatrix(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols, Type type = Replicate, T padValue = 0);

    // Construct a vector containing multidistance cv::Mats from a complete set of data
    std::vector<cv::Mat> convertVecToMats(const std::vector<uint16_t> &data, int numImages, int rows, int cols);
    F2DArray convertMatsToVec(const std::vector<cv::Mat> &mats);
    void removeOutliers(cv::Mat &originalImg, int kernelSize = 3, float threshold = 2.0f);

    cv::Mat xcorr2(const cv::Mat& A, const cv::Mat& B);
    cv::Mat filterImage(const cv::Mat &image, int kernelSize = 3, float stddev = 0.0f);
    // Align 2 images and returns the shifted pixels between the images
    cv::Point2f alignImages(cv::Mat &imAli, const cv::Mat &imRef, bool applyFilter = false, int kernelSize = 3);
    // matchTemplate function
    cv::Point2f alignImages01(cv::Mat &imAli, const cv::Mat &imRef, bool applyFilter = false);
    
    cv::Mat genCorrMatrix(const cv::Mat &image, int range, int windowSize);
    // Remove stripes of different dimensions
    void removeStripes(cv::Mat &image, int rangeRows = 0, int rangeCols = 0, int windowSize = 5, const std::string &method = "multiplication");

    void displayNDArray(F2DArray &images, int rows, int cols, const std::vector<std::string> &imgName);
}

template<class T>
std::vector<T> ImageUtils::cropMatrix(const std::vector<T> &matrix, int rows, int cols, int cropPreRows,
                                      int cropPreCols, int cropPostRows, int cropPostCols)
{
    if (cropPreRows < 0 || cropPreCols < 0 || cropPostRows < 0 || cropPostCols < 0) {
        throw std::invalid_argument("Cropping size cannot be less than 0!");
    }

    // Calculate the bounds for cropping
    int startRow = cropPreRows;
    int endRow = rows - cropPostRows;
    int startCol = cropPreCols;
    int endCol = cols - cropPostCols;

    if (startRow >= endRow || startCol >= endCol) {
        throw std::invalid_argument("Invalid cropping size!");
    }
    
    size_t size = (rows - cropPreRows - cropPostRows) * (cols - cropPreCols - cropPostCols);
    // Create a new cropped matrix
    std::vector<T> croppedMatrix(size);
    size_t num = 0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = startCol; j < endCol; ++j) {
            croppedMatrix[num++] = matrix[i * cols + j];
        }
    }

    return croppedMatrix;
}

template<class T>
std::vector<T> ImageUtils::padByConstant(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols, T padValue)
{
    // Calculate new dimensions
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;

    // Initialize the padded array with the padValue
    std::vector<T> paddedMatrix(newRows * newCols, padValue);

    // Copy the original array into the center of the padded array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            paddedMatrix[(i + padRows) * newCols + (j + padCols)] = matrix[i * cols + j];
        }
    }

    return paddedMatrix;
}

template<class T>
std::vector<T> ImageUtils::padByReplicate(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols)
{
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;
    std::vector<T> paddedMatrix(newRows * newCols);

    // Pad center part
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            paddedMatrix[(i + padRows) * newCols + (j + padCols)] = matrix[i * cols + j];
        }
    }

    // Pad upper and lower boundry
    for (int j = 0; j < cols; ++j) {
        for (int k = 0; k < padRows; ++k) {
            paddedMatrix[k * newCols + (j + padCols)] = matrix[j];
            paddedMatrix[(rows + padRows + k) * newCols + (j + padCols)] = matrix[(rows - 1) * cols + j];
        }
    }

    // Pad left and right boundry
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < padCols; ++k) {
            paddedMatrix[(i + padRows) * newCols + k] = matrix[i * cols];
            paddedMatrix[(i + padRows) * newCols + (cols + padCols + k)] = matrix[i * cols + (cols - 1)];
        }
    }

    // Pad 4 corners
    for (int i = 0; i < padRows; ++i) {
        for (int j = 0; j < padCols; ++j) {
            paddedMatrix[i * newCols + j] = matrix[0];
            paddedMatrix[i * newCols + (newCols - padCols + j)] = matrix[cols - 1];
            paddedMatrix[(newRows - padRows + i) * newCols + j] = matrix[(rows - 1) * cols];
            paddedMatrix[(newRows - padRows + i) * newCols + (newCols - padCols + j)] = matrix[(rows - 1) * cols + (cols - 1)];
        }
    }

    return paddedMatrix;
}

template<class T>
std::vector<T> ImageUtils::padMatrix(const std::vector<T> &matrix, int rows, int cols, int padRows, int padCols, Type type, T padValue)
{
    if (padRows < 0 || padCols < 0) {
        throw std::invalid_argument("Padding size cannot be less than 0!");
    }
    
    std::vector<T> paddedMatrix;
    switch (type)
    {
        case Constant:
            paddedMatrix = padByConstant(matrix, rows, cols, padRows, padCols, padValue);
            break; 
        case Replicate:
            paddedMatrix = padByReplicate(matrix, rows, cols, padRows, padCols);
            break;
        default:
            throw std::invalid_argument("Invalid padding method!");
    }

    return paddedMatrix;
}

namespace IOUtils
{   
    bool readRawData(const std::string &filename, const std::string &datasetName, std::vector<uint16_t> &data, std::vector<hsize_t> &dims);
    bool readPhasegrams(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims);
    bool readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims);
    bool saveProcessedGrams(const std::string &filename, const std::string &datasetName, const FArray &processedGrams, int numImages, int rows, int cols);
    bool savePhasegrams(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols);

}

#endif