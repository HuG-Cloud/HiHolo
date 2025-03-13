#include <iostream>
#include <string>
#include "imageio_utils.h"

const int numImages = 4;

int main()
{
    const std::string inputPath = "/home/hug/Downloads/HoloTomo_Data/result_";
    DArray nz {0.3, 0.5, 0.7, 0.9};

    for (int i = 0; i < numImages; i++)
    {
        cv::Mat colorImage = cv::imread(inputPath + std::to_string(i) + ".tif", cv::IMREAD_UNCHANGED);

        cv::Mat grayImage;
        if (colorImage.channels() > 1) {
            cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
        }
        else {
            grayImage = colorImage.clone();
        }

        cv::Mat grayImg32F;
        grayImage.convertTo(grayImg32F, CV_32F);

        double npixels = ImageUtils::computePixels(grayImg32F);
        std::cout << "Vmax (n=" << nz[i] << "): " << 1.0 / npixels << std::endl;
    }

    return 0;
}