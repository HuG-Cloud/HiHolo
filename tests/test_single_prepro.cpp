#include <chrono>
#include <iostream>

#include "io_utils.h"
#include "image_utils.h"

int main(int argc, char* argv[]) {
    const std::string inputPath1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board.h5";
    const std::string inputPath2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_back.h5";
    const std::string outputPath1 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_holo.h5";
    const std::string outputPath2 = "/home/hug/Downloads/HoloTomo_Data/visiblelight/board_probe.h5";
    const std::string datasetName = "holodata";
   
    std::vector<hsize_t> dims;
    bool isAPWP = false;
   
    FArray holograms, probes;
    IOUtils::readPhaseGram(inputPath1, datasetName, holograms, dims);
    IOUtils::readPhaseGram(inputPath2, datasetName, probes, dims);
    int rows = static_cast<int>(dims[0]);
    int cols = static_cast<int>(dims[1]);
    int numImages = 1;

    cv::Mat holoMat = ImageUtils::convertVecToMat(holograms, rows, cols);
    cv::Mat probeMat = ImageUtils::convertVecToMat(probes, rows, cols);
    ImageUtils::removeOutliers(holoMat);
    ImageUtils::removeOutliers(probeMat);

    if (!isAPWP) {
        holoMat /= probeMat;
    }
    
    cv::Rect cropRoi(300, 800, 2049, 2049);
    holoMat = holoMat(cropRoi).clone();
    probeMat = probeMat(cropRoi).clone();
    rows = holoMat.rows;
    cols = holoMat.cols;
    
    //ImageUtils::removeStripes(holoMat);
    FArray holoVec = ImageUtils::convertMatToVec(holoMat);
    //ImageUtils::displayPhase(holoVec, rows, cols, "holoVec");
    IOUtils::save3DGrams(outputPath1, datasetName, holoVec, numImages, rows, cols);
    ImageUtils::saveImage("holo_board.png", holoVec, rows, cols);
    
    if (isAPWP) {
        FArray probeVec = ImageUtils::convertMatToVec(probeMat);
        IOUtils::save3DGrams(outputPath2, datasetName, probeVec, numImages, rows, cols);
    }

    return 0;
}