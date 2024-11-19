#include "imageio_utils.h"

const uchar maxUChar = std::numeric_limits<uchar>::max();

std::vector<cv::Mat> ImageUtils::convertVecToMats(const std::vector<uint16_t> &data, int numImages, int rows, int cols)
{
    if (data.size() != static_cast<size_t>(numImages * rows * cols)) {
        throw std::invalid_argument("Data size does not match the specified number and dimension!");
    }

    std::vector<cv::Mat> mats(numImages, cv::Mat(rows, cols, CV_16U));
    for (int i = 0; i < numImages; i++) {
        memcpy(mats[i].data, data.data() + i * rows * cols, rows * cols * sizeof(uint16_t));
    }
    
    for (auto &mat: mats) {
        mat.convertTo(mat, CV_32F);
    }
    
    return mats;
}

// D2DArray ImageUtils::convertMatsToVec(const std::vector<cv::Mat> &mats)
// {
//     D2DArray grams;
//     for (const auto &mat: mats) {
//         // Check the mat is of type float
//         if (mat.type() != CV_32F) {
//             std::cerr << "Mat type is not CV_32F!" << std::endl;
//             return D2DArray();
//         }
        
//         DArray gram(mat.rows * mat.cols);
//         for (int i = 0; i < mat.rows; i++) {
//             for (int j = 0; j < mat.cols; j++) {
//                 gram[i * mat.cols + j] = static_cast<double>(mat.at<float>(i, j));
//             }
//         }
        
//         grams.push_back(gram);
//     }

//     return grams;
// }


void ImageUtils::removeOutliers(cv::Mat &originalImg, int kernelSize, float threshold)
{   
    // Set the zero value to max
    cv::MatIterator_<float> end = originalImg.end<float>();
    for (auto it = originalImg.begin<float>(); it != end; it++)
    {
        if (*it == 0 || *it == maxUInt_16) {
            *it = FloatInf;
        }
        
    }

    // Median filter
    cv::Mat filteredImg;
    cv::medianBlur(originalImg, filteredImg, kernelSize);
    cv::Mat differenceImg = originalImg - filteredImg;

    // Calculate the standard deviation of a finite value
    cv::Mat nonInfMask(differenceImg.size(), CV_8U);
    nonInfMask.setTo(maxUChar);
    for (int i = 0; i < differenceImg.rows; i++) {
        for (int j = 0; j < differenceImg.cols; j++) {
            if (std::isinf(differenceImg.at<float>(i, j))) {
                nonInfMask.at<uchar>(i, j) = 0;
            }
        }
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(differenceImg, mean, stddev, nonInfMask);

    // Pixels that need to be corrected
    cv::Mat absDiffImg;
    cv::absdiff(differenceImg, cv::Scalar::all(0), absDiffImg);
    cv::Mat biasedPixels = absDiffImg > threshold * stddev[0];

    filteredImg.copyTo(originalImg, biasedPixels);
}

cv::Mat ImageUtils::xcorr2(const cv::Mat& X, const cv::Mat& H)
{
    // Flip the kernel for cross-correlation
    cv::Mat H_flipped;
    cv::flip(H, H_flipped, -1);

    // Calculate the output size
    int rows = X.rows + H.rows - 1;
    int cols = X.cols + H.cols - 1;

    // Pad the input matrix
    cv::Mat X_padded;
    cv::copyMakeBorder(X, X_padded, H.rows - 1, H.rows - 1, H.cols - 1, H.cols - 1, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Perform the filtering
    cv::Mat C;
    cv::filter2D(X_padded, C, CV_32F, H_flipped, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    return C;
}

cv::Mat ImageUtils::filterImage(const cv::Mat &image, int kernelSize, float stddev)
{
    cv::Mat filteredImage;

    // Apply gaussian low-pass filter to remove noise
    cv::GaussianBlur(image, filteredImage, cv::Size(kernelSize, kernelSize), stddev);
    // Apply high-pass filter to enhance edge
    cv::Laplacian(filteredImage, filteredImage, CV_32F);
    cv::medianBlur(filteredImage, filteredImage, kernelSize);

    return filteredImage;
}

cv::Point2f ImageUtils::alignImages(cv::Mat &imAli, const cv::Mat &imRef, bool applyFilter, int kernelSize)
{
    if (imAli.size() != imRef.size()) {
        throw std::invalid_argument("The sizes of 2 images do not match!");
    }

    // Filter the 2 images before alignment
    cv::Mat filteredImAli = imAli.clone();
    cv::Mat filteredImRef = imRef.clone();

    if (applyFilter) {
        filteredImAli = filterImage(filteredImAli, kernelSize);
        filteredImRef = filterImage(filteredImRef, kernelSize);
    }

    // Computational cross correlation and finde the location of max value
    cv::Mat corrMatrix = xcorr2(filteredImAli, filteredImRef);
    std::cout << corrMatrix.size << std::endl;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(corrMatrix, &minVal, &maxVal, &minLoc, &maxLoc);
    
    // cv::Point2i center(filteredImAli.cols - 1, filteredImAli.rows - 1);
    // Calculate translation and transform
    cv::Point2f shift(maxLoc.x, maxLoc.y);
    cv::Mat transMatrix = (cv::Mat_<float>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(imAli, imAli, transMatrix, imAli.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    return shift;
}

cv::Point2f ImageUtils::alignImages01(cv::Mat &imAli, const cv::Mat &imRef, bool applyFilter)
{
    if (imAli.size() != imRef.size()) {
        throw std::invalid_argument("The sizes of 2 images do not match!");
    }

    // Filter the 2 images before alignment
    cv::Mat filteredImAli = imAli.clone();
    cv::Mat filteredImRef = imRef.clone();

    int padSize = imRef.rows / 2;
    cv::copyMakeBorder(filteredImRef, filteredImRef, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT, cv::Scalar(0));
    if (applyFilter) {
        filteredImAli = filterImage(filteredImAli);
        filteredImRef = filterImage(filteredImRef);
    }

    cv::Mat corrMatrix;
    cv::matchTemplate(filteredImRef, filteredImAli, corrMatrix, cv::TM_CCORR_NORMED);
    
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(corrMatrix, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point2f shift(maxLoc.x - padSize, maxLoc.y - padSize);
    cv::Mat transMatrix = (cv::Mat_<float>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(imAli, imAli, transMatrix, imAli.size());

    return shift;
}

cv::Mat ImageUtils::genCorrMatrix(const cv::Mat &image, int range, int windowSize)
{
    cv::Mat topImage, bottomImage, topCols, bottomCols;
    
    // Calculate the average intensity of the top rols
    topImage = image.rowRange(0, range);
    cv::reduce(topImage, topCols, 0, cv::REDUCE_AVG);
    cv::blur(topCols, topCols, cv::Size(windowSize, 1), cv::Point(-1 ,-1), cv::BORDER_REPLICATE);

    // Calculate the average intensity of the bottom rols
    bottomImage = image.rowRange(image.rows - range, image.rows);
    cv::reduce(bottomImage, bottomCols, 0, cv::REDUCE_AVG);
    cv::blur(bottomCols, bottomCols, cv::Size(windowSize, 1), cv::Point(-1 ,-1), cv::BORDER_REPLICATE);

    // Generate linear interpolation factor
    FArray linearRange = MathUtils::genEquidisRange(0, 1, image.rows);
    cv::Mat interFactor(image.rows, 1, CV_32F, linearRange.data());
    
    // Combine top and bottom intensity to remove stripes
    return (1 - interFactor) * topCols + interFactor * bottomCols;
}


void ImageUtils::removeStripes(cv::Mat &image, int rangeRows, int rangeCols, int windowSize, const std::string &method)
{
    if (rangeRows == 0) {
        rangeRows = std::ceil(image.rows * 0.04);
    }

    if (rangeCols == 0) {
        rangeCols = std::ceil(image.cols * 0.04);
    }
    
    // Create correction matrix for horizontal and vertical
    cv::Mat corrMatrix_y = genCorrMatrix(image, rangeRows, windowSize);
    cv::Mat corrMatrix_x = genCorrMatrix(image.t(), rangeCols, windowSize);
    auto corrMatrix = corrMatrix_y + corrMatrix_x.t();

    // Remove stripes by dividing or subtracting
    if (method == "multiplication") {
        image /= corrMatrix - cv::mean(corrMatrix)[0] + 1;
    } else if (method == "addition") {
        image -= corrMatrix - cv::mean(corrMatrix)[0];
    } else {
        throw std::invalid_argument("Invalid removal method!");
    }

}

void ImageUtils::displayNDArray(F2DArray &images, int rows, int cols, const std::vector<std::string> &imgName)
{   
    std::vector<cv::Mat> mats(images.size(), cv::Mat(rows, cols, CV_32F));
    for (int i = 0; i < mats.size(); i++) {
        memcpy(mats[i].data, images[i].data(), rows * cols * sizeof(float));
        cv::normalize(mats[i], mats[i], 0, 255, cv::NORM_MINMAX);
        mats[i].convertTo(mats[i], CV_8U);
    }
    
    for (int i = 0; i < mats.size(); i++) {
        cv::namedWindow(imgName[i], cv::WINDOW_AUTOSIZE);
    }

    for (int i = 0; i < mats.size(); i++) {
        cv::imshow(imgName[i], mats[i]);
    }
    while (cv::waitKey(1) != 27);    
}


// bool IOUtils::readRawData(const std::string &filename, const std::string &datasetName, std::vector<uint16_t> &data, std::vector<hsize_t> &dims)
// {
//     try {
//         // Open H5 file and dataset
//         H5::H5File file(filename, H5F_ACC_RDONLY);
//         H5::DataSet dataset = file.openDataSet(datasetName);

//         // Make sure the dataset dimensions is 3D
//         H5::DataSpace dataspace = dataset.getSpace();
//         int rank = dataspace.getSimpleExtentNdims();
//         if (rank != 3) {
//             std::cerr << "Error: DataSet is not 3-dimensional!" << std::endl;
//             return false;
//         }

//         dims.resize(rank);
//         dataspace.getSimpleExtentDims(dims.data(), nullptr);

//         data.resize(dims[0] * dims[1] * dims[2]);
//         dataset.read(data.data(), H5::PredType::NATIVE_UINT16);

//         return true;
//     } catch(H5::Exception &error) {
//         std::cerr << "Error reading dataset: " << error.getDetailMsg() << std::endl;
//         return false;
//     }
    
// }

bool IOUtils::readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims)
{       
    try {
        // Open H5 file and dataset
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);

        // Make sure the dataset dimensions is 3D
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 3) {
            std::cerr << "Error: DataSet is not 3-dimensional!" << std::endl;
            return false;
        }

        dims.resize(rank);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);
        holograms.resize(dims[0] * dims[1] * dims[2]);
        dataset.read(holograms.data(), H5::PredType::NATIVE_FLOAT);

    } catch(H5::Exception &error) {
        std::cerr << "Error reading dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }

    return true;
}

bool IOUtils::readPhasegrams(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims)
{
    try {
        // Open H5 file and dataset
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);

        // Make sure the dataset dimensions is 2D
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 2) {
            std::cerr << "Error: DataSet is not 2-dimensional!" << std::endl;
            return false;
        }

        dims.resize(rank);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);

        phase.resize(dims[0] * dims[1]);
        dataset.read(phase.data(), H5::PredType::NATIVE_FLOAT);

        return true;
    } catch(H5::Exception &error) {

        std::cerr << "Error reading dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
}

bool IOUtils::savePhasegrams(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols)
{
    try {
        // Create file and data space
        H5::H5File file(filename, H5F_ACC_TRUNC);
        hsize_t dims[2] {rows, cols};
        H5::DataSpace dataspace(2, dims);

        // Create dataset and write data
        H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(reconsPhase.data(), H5::PredType::NATIVE_FLOAT);

        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error writing dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
    
}

// bool IOUtils::saveProcessedGrams(const std::string &filename, const std::string &datasetName, const FArray &processedGrams, int numImages, int rows, int cols)
// {
//     // Converts 2-dimensional vector to 1-dimensional vector
//     DArray flatData;
//     for (const auto &row: processedGrams) {
//         flatData.insert(flatData.end(), row.begin(), row.end());
//     }
    
//     try {
//         // Create file and data space
//         H5::H5File file(filename, H5F_ACC_TRUNC);
//         hsize_t dims[3] {processedGrams.size(), rows, cols};
//         H5::DataSpace dataspace(3, dims);

//         // Create dataset and write data
//         H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE, dataspace);
//         dataset.write(flatData.data(), H5::PredType::NATIVE_DOUBLE);

//         return true;
//     } catch(H5::Exception &error) {
//         std::cerr << "Error writing dataset: " << error.getDetailMsg() << std::endl;
//         return false;
//     }
// }