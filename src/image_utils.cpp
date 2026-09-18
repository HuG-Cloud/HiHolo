#include "image_utils.h"

const uchar maxUChar = std::numeric_limits<uchar>::max();

std::vector<cv::Mat> ImageUtils::convertVecToMats(const U16Array &data, int numImages, int rows, int cols)
{
    std::vector<cv::Mat> mats(numImages, cv::Mat(rows, cols, CV_16U));
    for (int i = 0; i < numImages; i++) {
        memcpy(mats[i].data, data.data() + i * rows * cols, rows * cols * sizeof(uint16_t));
        mats[i].convertTo(mats[i], CV_32F);
    }
    
    return mats;
}

std::vector<cv::Mat> ImageUtils::convertVecToMats(const FArray &data, int numImages, int rows, int cols)
{
    std::vector<cv::Mat> mats(numImages, cv::Mat(rows, cols, CV_32F));
    for (int i = 0; i < numImages; i++) {
        memcpy(mats[i].data, data.data() + i * rows * cols, rows * cols * sizeof(float));
    }
    
    return mats;
}

cv::Mat ImageUtils::convertVecToMat(const U16Array &data, int rows, int cols)
{
    cv::Mat mat(rows, cols, CV_16U);
    memcpy(mat.data, data.data(), rows * cols * sizeof(uint16_t));
    mat.convertTo(mat, CV_32F);

    return mat;
}

cv::Mat ImageUtils::convertVecToMat(const FArray &data, int rows, int cols)
{
    cv::Mat mat(rows, cols, CV_32F);
    memcpy(mat.data, data.data(), rows * cols * sizeof(float));

    return mat;
}

FArray ImageUtils::convertMatsToVec(const std::vector<cv::Mat> &mats)
{
    FArray grams(mats.size() * mats[0].rows * mats[0].cols);
    for (int i = 0; i < mats.size(); i++) {
        memcpy(grams.data() + i * mats[i].rows * mats[i].cols,
               mats[i].data, mats[i].rows * mats[i].cols * sizeof(float));
    }

    return grams;
}

FArray ImageUtils::convertMatToVec(const cv::Mat &mat)
{
    FArray data(mat.rows * mat.cols);
    memcpy(data.data(), mat.data, mat.rows * mat.cols * sizeof(float));
    return data;
}

void ImageUtils::removeOutliers(cv::Mat &originalImg, int kernelSize, float threshold)
{   
    // Set the zero value to max
    cv::MatIterator_<float> end = originalImg.end<float>();
    for (auto it = originalImg.begin<float>(); it != end; it++)
    {
        if (*it == 0 || *it == maxUInt_16 || std::isnan(*it)) {
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

cv::Mat ImageUtils::genCorrMatrix(const cv::Mat &image, int range, int windowSize)
{
    cv::Mat topImage, bottomImage, topCols, bottomCols;
    
    // Calculate the average intensity of the top rows
    topImage = image.rowRange(0, range);
    cv::reduce(topImage, topCols, 0, cv::REDUCE_AVG);
    cv::blur(topCols, topCols, cv::Size(windowSize, 1), cv::Point(-1 ,-1), cv::BORDER_REPLICATE);

    // Calculate the average intensity of the bottom rows
    bottomImage = image.rowRange(image.rows - range, image.rows);
    cv::reduce(bottomImage, bottomCols, 0, cv::REDUCE_AVG);
    cv::blur(bottomCols, bottomCols, cv::Size(windowSize, 1), cv::Point(-1 ,-1), cv::BORDER_REPLICATE);

    // Ensure the columns have finite values
    cv::Mat topColsMask, bottomColsMask;
    cv::compare(topCols, topCols, topColsMask, cv::CMP_EQ);  // Check for NaN
    cv::compare(bottomCols, bottomCols, bottomColsMask, cv::CMP_EQ);  // Check for NaN
    
    // Replace invalid values with image mean
    cv::Scalar imgMean = cv::mean(image);
    topCols.setTo(imgMean[0], ~topColsMask);
    bottomCols.setTo(imgMean[0], ~bottomColsMask);

    // Generate linear interpolation factor
    FArray linearRange = MathUtils::genEquidisRange(0, 1, image.rows);
    cv::Mat interFactor(image.rows, 1, CV_32F, linearRange.data());
    
    // Create correction matrix by proper interpolation
    cv::Mat corrMatrix(image.rows, image.cols, CV_32F);
    
    // Expand topCols and bottomCols to full image size
    cv::Mat topColsExpanded, bottomColsExpanded;
    cv::repeat(topCols, image.rows, 1, topColsExpanded);
    cv::repeat(bottomCols, image.rows, 1, bottomColsExpanded);
    
    // Expand interpolation factor to match image width
    cv::Mat interFactorExpanded;
    cv::repeat(interFactor, 1, image.cols, interFactorExpanded);
    
    // Linear interpolation between top and bottom
    corrMatrix = (1.0 - interFactorExpanded).mul(topColsExpanded) + 
                 interFactorExpanded.mul(bottomColsExpanded);
    
    return corrMatrix;
}


void ImageUtils::removeStripes(cv::Mat &image, int rangeRows, int rangeCols, int windowSize, const std::string &method)
{
    if (rangeRows == 0) {
        rangeRows = std::ceil(image.rows * 0.04);
    }

    if (rangeCols == 0) {
        rangeCols = std::ceil(image.cols * 0.04);
    }
    
    // Ensure ranges are valid
    rangeRows = std::max(1, std::min(rangeRows, image.rows / 4));
    rangeCols = std::max(1, std::min(rangeCols, image.cols / 4));
    
    // Ensure window size is odd and positive
    if (windowSize <= 0 || windowSize % 2 == 0) {
        windowSize = 5;  // Default to 5
    }
    
    try {
        // Create correction matrix for horizontal and vertical stripes
        cv::Mat corrMatrix_y = genCorrMatrix(image, rangeRows, windowSize);
        cv::Mat corrMatrix_x = genCorrMatrix(image.t(), rangeCols, windowSize);
        cv::Mat corrMatrix = corrMatrix_y + corrMatrix_x.t();

        // Normalize the correction matrix to avoid extreme values
        cv::Scalar meanCorr = cv::mean(corrMatrix);
        double meanVal = meanCorr[0];
        
        // Check for invalid mean value
        if (!std::isfinite(meanVal) || meanVal <= 0) {
            std::cerr << "Warning: Invalid correction matrix mean, skipping stripe removal" << std::endl;
            return;
        }
        
        // Remove stripes by dividing or subtracting
        if (method == "mul") {
            // For multiplicative correction, normalize to mean=1
            cv::Mat normalizedCorr = corrMatrix / meanVal;
            
            // Add small epsilon to avoid division by zero or very small numbers
            const double epsilon = 1e-6;
            cv::Mat denominator;
            cv::max(normalizedCorr, epsilon, denominator);
            
            image = image / denominator;
        } else if (method == "add") {
            // For additive correction, subtract the offset pattern
            cv::Mat offsetPattern = corrMatrix - meanVal;
            image = image - offsetPattern;
        } else {
            throw std::invalid_argument("Invalid removal method! Use 'mul' or 'add'.");
        }
        
        // Check for invalid values in result and clamp if necessary
        cv::Mat mask;
        cv::compare(image, image, mask, cv::CMP_EQ);  // Check for NaN
        if (cv::countNonZero(~mask) > 0) {
            std::cerr << "Warning: NaN values detected after stripe removal, clamping to zero" << std::endl;
            image.setTo(0, ~mask);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in stripe removal: " << e.what() << std::endl;
        throw;
    }
}

DArray ImageUtils::computePSDs(const std::vector<cv::Mat> &images, int direction, std::vector<cv::Mat> &profiles, std::vector<cv::Mat> &frequencies)
{
    DArray maxFre(images.size());
    for (int i = 0; i < images.size(); i++) {
        maxFre[i] = computePSD(images[i], direction, profiles[i], frequencies[i]);
    }

    return maxFre;
}

D2DArray ImageUtils::calibrateDistance(const DArray &maxPSD, const DArray &nz, double length, double pixelSize, double stepSize)
{
    // Compute pixels and magnification
    int numImages = maxPSD.size();
    DArray magnitudes(numImages);
    const long double magnitudeScale = static_cast<long double>(length) / static_cast<long double>(pixelSize);
    for (int i = 0; i < numImages; i++) {
        magnitudes[i] = static_cast<double>(static_cast<long double>(maxPSD[i]) * magnitudeScale);
    }

    // Fit the 1/M and nz to a linear function
    double a, b;
    double cov[4];
    gsl_fit_linear(nz.data(), 1, magnitudes.data(), 1, numImages, &a, &b, cov, cov + 1, cov + 2, cov + 3);

    // Return the distances and the fitting parameters
    D2DArray parameters(4);
    parameters[0] = nz;
    parameters[1] = magnitudes;

    DArray mag_fits(numImages);
    for (int i = 0; i < numImages; i++) {
        mag_fits[i] = std::fma(b, nz[i], a);
    }
    parameters[2] = mag_fits;

    // Compute the distances
    const long double z2 = static_cast<long double>(stepSize) / static_cast<long double>(b);
    const long double z1 = static_cast<long double>(a) * z2;
    parameters[3] = {z1, z2, b, a, cov[3]};

    return parameters;
}

double ImageUtils::computePSD(const cv::Mat &image, int direction, cv::Mat &profile, cv::Mat &fre)
{
    // Split the image into real and imaginary parts
    cv::Mat planes[] = {image, cv::Mat::zeros(image.size(), image.type())};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);

    // Compute PSD
    cv::split(complexImg, planes);
    cv::Mat psd;
    cv::magnitude(planes[0], planes[1], psd);
    cv::pow(psd, 2, psd);

    // Sum along the column or row direction
    // 0 for column, 1 for row
    cv::reduce(psd, profile, direction, cv::REDUCE_SUM, CV_64F);

    int size;
    if (direction == 0) { // 如果沿列方向归约，使用列数
        size = image.cols;
    } else { // 如果沿行方向归约，使用行数
        size = image.rows;
    }
    
    // 创建频率向量，只取一半（到奈奎斯特频率），作为行向量
    fre = cv::Mat(1, size / 2, CV_64F);
    double* freData = reinterpret_cast<double*>(fre.data);
    for (int i = 0; i < size / 2; i++) {
        freData[i] = static_cast<double>(i) / static_cast<double>(size);
    }
    
    if (profile.rows > 1) {
        profile = profile.t();
    }
    
    // 截取profile以匹配频率轴的大小
    if (profile.total() > fre.total()) {
        profile = profile.colRange(0, fre.cols);
    }

    // 去掉第一个点
    profile = profile.colRange(1, profile.cols);
    fre = fre.colRange(1, fre.cols);

    // Compute the maximum value and its index (coordinate)
    cv::Point maxLoc;
    cv::minMaxLoc(profile, nullptr, nullptr, nullptr, &maxLoc);

    // profile is a row vector
    int maxIndex = maxLoc.x;
    double maxFre = fre.at<double>(0, maxIndex);

    return maxFre;
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

void ImageUtils::displayPhase(FArray &phase, int rows, int cols, const std::string &imgName)
{   
    // Create OpenCV matrix
    cv::Mat mat(rows, cols, CV_32F);
    memcpy(mat.data, phase.data(), rows * cols * sizeof(float));
    
    // Normalize to 0-255 range and convert to 8-bit unsigned integer
    cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    mat.convertTo(mat, CV_8U);
    
    // Use a fixed window name but display different titles
    static const std::string windowName = "Phase Display";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setWindowTitle(windowName, imgName);
    cv::imshow(windowName, mat);
    cv::waitKey(3000);
}

bool ImageUtils::saveImage(const std::string &filename, const FArray &image, int rows, int cols)
{
    // 创建OpenCV矩阵
    cv::Mat mat(rows, cols, CV_32F);
    memcpy(mat.data, image.data(), rows * cols * sizeof(float));
    
    // 归一化到0-255范围并转换为8位无符号整数
    cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    mat.convertTo(mat, CV_8U);
    
    // 保存为jpg格式图片
    bool success = cv::imwrite(filename, mat);
    if (!success) {
        std::cerr << "保存图片失败: " << filename << std::endl;
        return false;
    }
    
    return true;
}
