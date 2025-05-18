#include "imageio_utils.h"

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

cv::Mat ImageUtils::convertVecToMat(const U16Array &data, int rows, int cols)
{
    cv::Mat mat(rows, cols, CV_16U);
    memcpy(mat.data, data.data(), rows * cols * sizeof(uint16_t));
    mat.convertTo(mat, CV_32F);

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

void ImageUtils::convertVecToImgs(float *data, std::vector<itk::simple::Image> &images, int rows, int cols)
{
    std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};
    for (int i = 0; i < images.size(); i++) {
        // 使用ImportImageFilter创建图像
        images[i] = itk::simple::ImportAsFloat(data + i * rows * cols, size, std::vector<double>(2, 1.0));
    }
}

void ImageUtils::convertMatsToImgs(const std::vector<cv::Mat> &mats, std::vector<itk::simple::Image> &images, int rows, int cols)
{
    std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};

    for (int i = 0; i < mats.size(); i++) {
        images[i] = itk::simple::ImportAsFloat(reinterpret_cast<float*>(mats[i].data), size, std::vector<double>(2, 1.0));
    }
}

void ImageUtils::convertImgsToVec(const std::vector<itk::simple::Image> &images, float *data, int rows, int cols)
{
    for (int i = 0; i < images.size(); i++) {
        auto buffer = images[i].GetBufferAsFloat();
        std::memcpy(data + i * rows * cols, buffer, rows * cols * sizeof(float));
    }
}

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

IntArray ImageUtils::registerImage(const itk::simple::Image &fixedImage, itk::simple::Image &movingImage)
{
    try {
        // Create registration method for image alignment
        itk::simple::ImageRegistrationMethod registration;
        
        // Configure registration parameters:
        // - Use correlation metric for similarity measure
        // - Gradient descent optimizer with specified parameters
        registration.SetMetricAsCorrelation();
        registration.SetOptimizerAsGradientDescent(1, 100, 1e-3, 15, registration.EachIteration);
        registration.SetOptimizerScalesFromPhysicalShift();
        
        // Set initial translation transform and linear interpolation
        registration.SetInitialTransform(itk::simple::TranslationTransform(fixedImage.GetDimension()));
        registration.SetInterpolator(itk::simple::sitkLinear);

        // Execute registration and adjust transform parameters
        itk::simple::Transform transform = registration.Execute(fixedImage, movingImage);
        DArray parameters = transform.GetParameters();
        if (std::abs(parameters[0]) > 1) {
            parameters[0] += parameters[0] > 0 ? 1 : -1;
        }
        if (std::abs(parameters[1]) > 1) {
            parameters[1] += parameters[1] > 0 ? 1 : -1;
        }

        // Pad image boundaries to accommodate translation
        std::vector<unsigned int> padBound = {static_cast<unsigned int>(std::round(std::abs(parameters[0]))),
                                              static_cast<unsigned int>(std::round(std::abs(parameters[1])))};
        movingImage = itk::simple::ZeroFluxNeumannPad(movingImage, padBound, padBound);

        // Apply transform and extract registered region
        transform.SetParameters(parameters);
        movingImage = itk::simple::Resample(movingImage, transform, itk::simple::sitkNearestNeighbor,
                                            0.0, movingImage.GetPixelID());
        IntArray index = {static_cast<int>(padBound[0]), static_cast<int>(padBound[1])};
        movingImage = itk::simple::Extract(movingImage, fixedImage.GetSize(), index);

        return {static_cast<int>(std::round(parameters[0])), static_cast<int>(std::round(parameters[1]))};
    } catch (const std::exception &e) {
        std::cerr << "Error registering image: " << e.what() << std::endl;
        return IntArray();
    }
}

D2DArray ImageUtils::calibrateDistance(const FArray &holograms, int numImages, int rows, int cols,
                                       double length, double pixelSize, const DArray &nz, double stepSize)
{
    // Convert holograms to cv::Mat
    std::vector<cv::Mat> holoMats(numImages, cv::Mat(rows, cols, CV_32F));
    for (int i = 0; i < numImages; i++) {
        memcpy(holoMats[i].data, holograms.data() + i * rows * cols, rows * cols * sizeof(float));
    }

    // Compute pixels and magnification
    DArray maxPSD(numImages), magnitudes(numImages);
    for (int i = 0; i < numImages; i++) {
        maxPSD[i] = computePSD(holoMats[i]);
        magnitudes[i] = 1.0 / ((1.0 / maxPSD[i]) * pixelSize / length);
    }

    // Fit the 1/M and nz to a linear function
    double a, b;
    double cov[4];
    gsl_fit_linear(nz.data(), 1, magnitudes.data(), 1, numImages, &a, &b, cov, cov + 1, cov + 2, cov + 3);

    // Return the distances and the fitting parameters
    D2DArray parameters(3);
    parameters[0] = maxPSD;
    parameters[1] = magnitudes;

    // Compute the distances
    double z1 = a * stepSize / b;
    double z2 = stepSize / b;
    parameters[2] = {b, a, cov[3], z1, z2};

    return parameters;
}

double ImageUtils::computePSD(const cv::Mat &image)
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

    // Sum along the column direction
    cv::Mat profile;
    cv::reduce(psd, profile, 0, cv::REDUCE_SUM, CV_32F);

    // Compute the number of pixels
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
    cv::waitKey(2000);
}

bool IOUtils::readRawData(const std::string &filename, const std::vector<std::string> &datasetNames,
                          std::vector<hsize_t> &dims, U16Array &data, U16Array &dark, U16Array &flat)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    
    // 打开H5文件
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    
    // 首先读取主数据集的维度信息
    dataset_id = H5Dopen2(file_id, datasetNames[0].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetNames[0] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetNames[0] << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    if (ndims != 3) {
        std::cerr << "Error: data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    dims.resize(ndims);
    status = H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions of dataspace" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    // 计算数据总大小
    hsize_t total_size = dims[0] * dims[1] * dims[2];
    
    // 分配内存并读取主数据
    data.resize(total_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error reading dataset: " << datasetNames[0] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 读取dark数据集
    dataset_id = H5Dopen2(file_id, datasetNames[1].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dark dataset: " << datasetNames[1] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (H5Sget_simple_extent_ndims(dataspace_id) != 3) {
        std::cerr << "Error: dark data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    std::vector<hsize_t> dark_dims(ndims);
    H5Sget_simple_extent_dims(dataspace_id, dark_dims.data(), nullptr);
    
    // 计算dark数据大小
    hsize_t dark_size = dark_dims[0] * dark_dims[1] * dark_dims[2];
    
    dark.resize(dark_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dark.data());
    if (status < 0) {
        std::cerr << "Error reading dark dataset: " << datasetNames[1] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 读取flat数据集
    dataset_id = H5Dopen2(file_id, datasetNames[2].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening flat dataset: " << datasetNames[2] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (H5Sget_simple_extent_ndims(dataspace_id) != 3) {
        std::cerr << "Error: flat data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    std::vector<hsize_t> flat_dims(ndims);
    H5Sget_simple_extent_dims(dataspace_id, flat_dims.data(), nullptr);
    
    // 计算flat数据大小
    hsize_t flat_size = flat_dims[0] * flat_dims[1] * flat_dims[2];
    
    flat.resize(flat_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    if (status < 0) {
        std::cerr << "Error reading flat dataset: " << datasetNames[2] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 关闭文件
    H5Fclose(file_id);
    return true;
}

bool IOUtils::readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间和维度
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    // 调整dims向量大小并获取维度信息
    dims.resize(ndims);
    status = H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions of dataspace for dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 确保数据集维度是3D
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 3) {
        std::cerr << "Error: DataSet is not 3-dimensional!" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取维度信息并读取数据到FArray
    dims.resize(rank);
    H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

    holograms.resize(dims[0] * dims[1] * dims[2]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, holograms.data());

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::readPhasegrams(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims)
{
    // 打开H5文件和数据集
    hid_t file_id, dataset_id, dataspace_id;
    
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 确保数据集维度是2D
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 2) {
        std::cerr << "Error: DataSet is not 2-dimensional!" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取维度信息并读取数据到FArray
    dims.resize(rank);
    H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

    phase.resize(dims[0] * dims[1]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phase.data());

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::savePhasegrams(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 创建文件
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return false;
    }

    // 创建数据空间
    hsize_t dims[2] {rows, cols};
    dataspace_id = H5Screate_simple(2, dims, nullptr);
    if (dataspace_id < 0) {
        std::cerr << "Error creating dataspace" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 创建数据集
    dataset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error creating dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reconsPhase.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::save3DGrams(const std::string &filename, const std::string &datasetName, const FArray &registeredGrams, int numImages, int rows, int cols)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 创建文件
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return false;
    }

    // 创建数据空间
    hsize_t dims[3] {numImages, rows, cols};
    dataspace_id = H5Screate_simple(3, dims, nullptr);
    if (dataspace_id < 0) {
        std::cerr << "Error creating dataspace" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 创建数据集
    dataset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error creating dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, registeredGrams.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return true;
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

bool IOUtils::read3DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id, memspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取数据维度
    hsize_t dims[3]; 
    status = H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 设置读取区域
    hsize_t offset_[3] = {offset, 0, 0};
    hsize_t count_[3] = {count, dims[1], dims[2]};
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset_, nullptr, count_, nullptr);

    // 创建内存空间
    memspace_id = H5Screate_simple(3, count_, nullptr);

    // 读取数据
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error reading dataset" << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::read4DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id, memspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取数据维度
    hsize_t dims[4]; 
    status = H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 设置读取区域
    hsize_t offset_[4] = {offset, 0, 0, 0};
    hsize_t count_[4] = {count, dims[1], dims[2], dims[3]};
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset_, nullptr, count_, nullptr);

    // 创建内存空间
    memspace_id = H5Screate_simple(4, count_, nullptr);

    // 调整数据大小并读取
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error reading dataset" << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::createFileDataset(const std::string &filename, const std::string &datasetName, const std::vector<hsize_t> &dims)
{
    // 使用C接口替代C++接口，以便更好地控制资源
    hid_t file_id, dataset_id, dataspace_id;

    // 创建HDF5文件
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return false;
    }

    // 创建数据空间
    dataspace_id = H5Screate_simple(dims.size(), dims.data(), nullptr);
    if (dataspace_id < 0) {
        std::cerr << "Error creating dataspace" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 创建数据集
    dataset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace_id, 
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error creating dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::write3DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id, memspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 计算要写入的数据大小
    hsize_t count_[3] = {data.size() / (dims[1] * dims[2]), dims[1], dims[2]};
    hsize_t offset_[3] = {offset, 0, 0};

    // 选择写入的区域
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset_, nullptr, count_, nullptr);
    // 创建内存空间
    memspace_id = H5Screate_simple(3, count_, nullptr);

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::write4DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id, memspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 计算要写入的数据大小
    hsize_t count_[4] = {data.size() / (dims[1] * dims[2] * dims[3]), dims[1], dims[2], dims[3]};
    hsize_t offset_[4] = {offset, 0, 0, 0};

    // 选择写入的区域
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset_, nullptr, count_, nullptr);
    // 创建内存空间
    memspace_id = H5Screate_simple(4, count_, nullptr);

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::saveImage(const std::string &filename, const FArray &image, int rows, int cols)
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