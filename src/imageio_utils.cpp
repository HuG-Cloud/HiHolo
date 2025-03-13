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

void ImageUtils::convertVecToImgs(float *data, std::vector<itk::simple::Image> &images, int rows, int cols)
{
    std::vector<unsigned int> size = {static_cast<unsigned int>(cols), static_cast<unsigned int>(rows)};

    for (int i = 0; i < images.size(); i++) {
        // 使用ImportImageFilter创建图像
        images[i] = itk::simple::ImportAsFloat(data + i * rows * cols, size, std::vector<double>(2, 1.0));
    }
}

void ImageUtils::convertImgsToVec(const std::vector<itk::simple::Image> &images, float *data, int rows, int cols)
{
    for (int i = 0; i < images.size(); i++) {
        auto buffer = images[i].GetBufferAsFloat();
        std::memcpy(data + i * rows * cols, buffer, rows * cols * sizeof(float));
    }
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

itk::simple::Transform ImageUtils::registerImage(const itk::simple::Image &fixedImage, itk::simple::Image &movingImage)
{
    try {
        // 创建配准器
        itk::simple::ImageRegistrationMethod registration;
        
        // 设置配准器参数
        registration.SetMetricAsCorrelation();
        registration.SetOptimizerAsGradientDescent(1, 100, 1e-3, 15, registration.EachIteration);
        registration.SetOptimizerScalesFromPhysicalShift();
        
        registration.SetInitialTransform(itk::simple::TranslationTransform(fixedImage.GetDimension()));
        registration.SetInterpolator(itk::simple::sitkLinear);

        // 执行配准
        itk::simple::Transform transform = registration.Execute(fixedImage, movingImage);
        std::vector<double> parameters = transform.GetParameters();
        if (std::abs(parameters[0]) > 1) {
            parameters[0] += parameters[0] > 0 ? 1 : -1;
        }
        if (std::abs(parameters[1]) > 1) {
            parameters[1] += parameters[1] > 0 ? 1 : -1;
        }

        std::vector<unsigned int> padBound = {static_cast<unsigned int>(std::round(std::abs(parameters[0]))),
                                              static_cast<unsigned int>(std::round(std::abs(parameters[1])))};
        movingImage = itk::simple::ZeroFluxNeumannPad(movingImage, padBound, padBound);

        transform.SetParameters(parameters);
        movingImage = itk::simple::Resample(movingImage, transform, itk::simple::sitkNearestNeighbor,
                                            0.0, movingImage.GetPixelID());
        IntArray index = {static_cast<int>(padBound[0]), static_cast<int>(padBound[1])};
        movingImage = itk::simple::Extract(movingImage, fixedImage.GetSize(), index);

        return transform;
    } catch (const std::exception &e) {
        std::cerr << "Error registering image: " << e.what() << std::endl;
        return itk::simple::Transform();
    }
}

DArray ImageUtils::calibrateDistance(const FArray &holograms, int numImages, int rows, int cols, double length,
                                     double pixelSize, const DArray &nz, double stepSize)
{
    // Convert holograms to cv::Mat
    std::vector<cv::Mat> holoMats(numImages, cv::Mat(rows, cols, CV_32F));
    for (int i = 0; i < numImages; i++) {
        memcpy(holoMats[i].data, holograms.data() + i * rows * cols, rows * cols * sizeof(float));
    }

    // Compute pixels and magnification
    DArray npixels(numImages), magnitudes(numImages);
    for (int i = 0; i < numImages; i++) {
        npixels[i] = computePixels(holoMats[i]);
        magnitudes[i] = 1.0 / (npixels[i] * pixelSize / length);
    }

    // Fit the 1/M and nz to a linear function
    double a, b;
    double cov[4];
    gsl_fit_linear(nz.data(), 1, magnitudes.data(), 1, numImages, &a, &b, cov, cov + 1, cov + 2, cov + 3);

    // Compute the distance
    DArray distances(2);
    distances[0] = a * stepSize / b;
    distances[1] = stepSize / b;

    return distances;
}

double ImageUtils::computePixels(const cv::Mat &image)
{
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

    return 1.0 / maxVal;
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

bool IOUtils::readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims)
{
    try {
        // 打开H5文件和数据集
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);

        // 获取数据空间和维度
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        
        // 调整dims向量大小并获取维度信息
        dims.resize(rank);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);

        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error reading dataset dimensions: " << error.getDetailMsg() << std::endl;
        return false;
    }
}

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

bool IOUtils::save3DGrams(const std::string &filename, const std::string &datasetName, const FArray &registeredGrams, int numImages, int rows, int cols)
{
    try {
        // Create file and data space
        H5::H5File file(filename, H5F_ACC_TRUNC);
        hsize_t dims[3] {numImages, rows, cols};
        H5::DataSpace dataspace(3, dims);

        // Create dataset and write data
        H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(registeredGrams.data(), H5::PredType::NATIVE_FLOAT);

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

bool IOUtils::read4DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count)
{
    try {
        // 打开HDF5文件和数据集
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);
        
        // 获取数据空间
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[4];
        dataspace.getSimpleExtentDims(dims, nullptr);
        
        // 设置读取区域
        hsize_t offset_[4] = {offset, 0, 0, 0};
        hsize_t count_[4] = {count, dims[1], dims[2], dims[3]};
        dataspace.selectHyperslab(H5S_SELECT_SET, count_, offset_);
        
        H5::DataSpace memspace(4, count_);        
        // 调整数据数组大小并读取数据
        data.resize(count_[0] * count_[1] * count_[2] * count_[3]);
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
        
        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error reading dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
}

bool IOUtils::createFileDataset(const std::string &filename, const std::string &datasetName, const std::vector<hsize_t> &dims)
{
    try {
        // 创建HDF5文件
        H5::H5File file(filename, H5F_ACC_TRUNC);
        // 创建数据空间
        H5::DataSpace dataspace(dims.size(), dims.data());
        // 创建数据集
        H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        
        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error creating dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
}

bool IOUtils::write3DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset)
{
    try {
        // 尝试打开文件，如果不存在则创建
        H5::H5File file;
        if (offset == 0) {
            file = H5::H5File(filename, H5F_ACC_TRUNC);
        } else {
            file.openFile(filename, H5F_ACC_RDWR);
        }
        
        H5::DataSet dataset;
        if (offset == 0) {
            // 如果数据集不存在，创建新的数据集
            H5::DataSpace dataspace(3, dims.data());
            dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_FLOAT, dataspace);
        } else {
            // 尝试打开已存在的数据集
            dataset = file.openDataSet(datasetName);
        }
        
        // 设置写入区域，一次写入一批数据
        hsize_t offset_[3] = {offset, 0, 0};
        hsize_t count_[3] = {data.size() / (dims[1] * dims[2]), dims[1], dims[2]};
        
        H5::DataSpace dataspace = dataset.getSpace();
        dataspace.selectHyperslab(H5S_SELECT_SET, count_, offset_);
        
        H5::DataSpace memspace(3, count_);        
        dataset.write(data.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
        
        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error writing dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
}

bool IOUtils::write4DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset)
{
    try {
        // 尝试打开文件，如果不存在则创建
        H5::H5File file = H5::H5File(filename, H5F_ACC_RDWR);        
        H5::DataSet dataset = file.openDataSet(datasetName);
        
        // 设置写入区域，一次写入一批数据
        hsize_t offset_[4] = {offset, 0, 0, 0};
        hsize_t count_[4] = {data.size() / (dims[1] * dims[2] * dims[3]), dims[1], dims[2], dims[3]};
        
        H5::DataSpace dataspace = dataset.getSpace();
        dataspace.selectHyperslab(H5S_SELECT_SET, count_, offset_);
        
        H5::DataSpace memspace(4, count_);        
        dataset.write(data.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
        
        return true;
    } catch(H5::Exception &error) {
        std::cerr << "Error writing dataset: " << error.getDetailMsg() << std::endl;
        return false;
    }
}