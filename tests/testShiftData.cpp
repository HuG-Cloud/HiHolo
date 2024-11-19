#include <iostream>
#include <random>
#include <chrono>

#include "../holo_recons.h"

int main()
{
    D2DArray shiftedHolograms;
    std::vector<hsize_t> dims;
    IOUtils::readProcessedGrams("/home/hug/Downloads/HoloTomo_Data/holo_purephase_shift.h5", "holodata", shiftedHolograms, dims);
    int rows = int(dims[1]);
    int cols = int(dims[2]);

    std::vector<cv::Mat> mats;
    for (int i = 0; i < dims[0]; i++) {
        mats.push_back(cv::Mat(rows, cols, CV_64F, shiftedHolograms[i].data()));
        mats[i].convertTo(mats[i], CV_32F);
    }
    
    // for (int i = 1; i < mats.size(); i++) {
    //     cv::Point2f shift = ImageUtils::alignImages(mats[i], mats[0], true);
    //     std::cout << shift << " ";
    // }
    int dx = 5;
    int dy = -8;
    cv::Mat newMat;
    cv::Mat transMat = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(mats[1], newMat, transMat, mats[0].size());
    cv::Point2f shift = ImageUtils::alignImages01(mats[0], mats[1], true);

    std::cout << shift << std::endl;

    // D2DArray holograms = ImageUtils::convertMatsToVec(mats);
    // D2DArray holograms;
    // IOUtils::readProcessedGrams("/home/hug/Downloads/HoloTomo_Data/holo_purephase_shift.h5", "holodata", holograms, dims);

    // ImageUtils::displayNDArray(holograms, rows, cols, {"1", "2", "3", "4"});

    // IOUtils::saveProcessedGrams("/home/hug/Downloads/HoloTomo_Data/holo_aligned_purephase01.h5", "holodata", holograms, rows, cols);
    // ImageUtils::displayNDArray(holograms, rows, cols, {"1", "2", "3", "4"});
    // IOUtils::readProcessedGrams("/home/hug/Downloads/HoloTomo_Data/holo_aligned_purephase.h5", "holodata", holograms, dims);
    // for (auto dim: dims) {
    //     std::cout << dim << " ";
    // }

    return 0;
}
