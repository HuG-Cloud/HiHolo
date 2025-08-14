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

// void ImageUtils::convertVecToImgs(float *data, std::vector<itk::simple::Image> &images, int rows, int cols)
// {
//     std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};
//     for (int i = 0; i < images.size(); i++) {
//         // 使用ImportImageFilter创建图像
//         images[i] = itk::simple::ImportAsFloat(data + i * rows * cols, size, std::vector<double>(2, 1.0));
//     }
// }

// void ImageUtils::convertMatsToImgs(const std::vector<cv::Mat> &mats, std::vector<itk::simple::Image> &images, int rows, int cols)
// {
//     std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};

//     for (int i = 0; i < mats.size(); i++) {
//         images[i] = itk::simple::ImportAsFloat(reinterpret_cast<float*>(mats[i].data), size, std::vector<double>(2, 1.0));
//     }
// }

// void ImageUtils::convertImgsToVec(const std::vector<itk::simple::Image> &images, float *data, int rows, int cols)
// {
//     for (int i = 0; i < images.size(); i++) {
//         auto buffer = images[i].GetBufferAsFloat();
//         std::memcpy(data + i * rows * cols, buffer, rows * cols * sizeof(float));
//     }
// }

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
    if (method == "mul") {
        image /= corrMatrix - cv::mean(corrMatrix)[0] + 1;
    } else if (method == "add") {
        image -= corrMatrix - cv::mean(corrMatrix)[0];
    } else {
        throw std::invalid_argument("Invalid removal method!");
    }

}

// IntArray ImageUtils::registerImage(const itk::simple::Image &fixedImage, itk::simple::Image &movingImage)
// {
//     try {
//         // Create registration method for image alignment
//         itk::simple::ImageRegistrationMethod registration;
        
//         // Configure registration parameters:
//         // - Use correlation metric for similarity measure
//         // - Gradient descent optimizer with specified parameters
//         registration.SetMetricAsCorrelation();
//         registration.SetOptimizerAsGradientDescent(1, 100, 1e-3, 15, registration.EachIteration);
//         registration.SetOptimizerScalesFromPhysicalShift();
        
//         // Set initial translation transform and linear interpolation
//         registration.SetInitialTransform(itk::simple::TranslationTransform(fixedImage.GetDimension()));
//         registration.SetInterpolator(itk::simple::sitkLinear);

//         // Execute registration and adjust transform parameters
//         itk::simple::Transform transform = registration.Execute(fixedImage, movingImage);
//         DArray parameters = transform.GetParameters();
//         if (std::abs(parameters[0]) > 1) {
//             parameters[0] += parameters[0] > 0 ? 1 : -1;
//         }
//         if (std::abs(parameters[1]) > 1) {
//             parameters[1] += parameters[1] > 0 ? 1 : -1;
//         }

//         // Pad image boundaries to accommodate translation
//         std::vector<unsigned int> padBound = {static_cast<unsigned int>(std::round(std::abs(parameters[0]))),
//                                               static_cast<unsigned int>(std::round(std::abs(parameters[1])))};
//         movingImage = itk::simple::ZeroFluxNeumannPad(movingImage, padBound, padBound);

//         // Apply transform and extract registered region
//         transform.SetParameters(parameters);
//         movingImage = itk::simple::Resample(movingImage, transform, itk::simple::sitkNearestNeighbor,
//                                             0.0, movingImage.GetPixelID());
//         IntArray index = {static_cast<int>(padBound[0]), static_cast<int>(padBound[1])};
//         movingImage = itk::simple::Extract(movingImage, fixedImage.GetSize(), index);

//         return {static_cast<int>(std::round(parameters[0])), static_cast<int>(std::round(parameters[1]))};
//     } catch (const std::exception &e) {
//         std::cerr << "Error registering image: " << e.what() << std::endl;
//         return IntArray();
//     }
// }

// Int2DArray ImageUtils::registerImages(float *data, int numImages, int rows, int cols)
// {
//     std::vector<itk::simple::Image> holoImages(numImages);
//     convertVecToImgs(data, holoImages, rows, cols);
//     Int2DArray translations(numImages);
//     translations[0] = {0, 0};

//     for (int i = 1; i < numImages; i++) {
//         translations[i] = ImageUtils::registerImage(holoImages[0], holoImages[i]);
//     }

//     return translations;
// }

DArray ImageUtils::computePSDs(const std::vector<cv::Mat> &images, int direction, std::vector<cv::Mat> &profiles, std::vector<cv::Mat> &frequencies)
{
    DArray maxPSD(images.size());
    for (int i = 0; i < images.size(); i++) {
        maxPSD[i] = computePSD(images[i], direction, profiles[i], frequencies[i]);
    }

    return maxPSD;
}

D2DArray ImageUtils::calibrateDistance(const DArray &maxPSD, const DArray &nz, double length, double pixelSize, double stepSize)
{
    // Compute pixels and magnification
    int numImages = maxPSD.size();
    DArray magnitudes(numImages);
    for (int i = 0; i < numImages; i++) {
        magnitudes[i] = 1.0 / ((1.0 / maxPSD[i]) * pixelSize / length);
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
        mag_fits[i] = b * nz[i] + a;
    }
    parameters[2] = mag_fits;

    // Compute the distances
    double z1 = a * stepSize / b;
    double z2 = stepSize / b;
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
    cv::reduce(psd, profile, direction, cv::REDUCE_SUM, CV_32F);

    int size;
    if (direction == 0) { // 如果沿列方向归约，使用列数
        size = image.cols;
    } else { // 如果沿行方向归约，使用行数
        size = image.rows;
    }
    
    // 创建频率向量，只取一半（到奈奎斯特频率），作为行向量
    fre = cv::Mat(1, size / 2, CV_32F);
    float* freData = reinterpret_cast<float*>(fre.data);
    for (int i = 0; i < size / 2; i++) {
        freData[i] = static_cast<float>(i) / static_cast<float>(size);
    }
    
    if (profile.rows > 1) {
        profile = profile.t();
    }
    
    // 截取profile以匹配频率轴的大小
    if (profile.total() > fre.total()) {
        profile = profile.colRange(0, fre.cols);
    }

    // Compute the maximum value
    double maxVal;
    cv::minMaxLoc(profile, nullptr, &maxVal);

    return maxVal;
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