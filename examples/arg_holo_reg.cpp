#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_registration");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-i")
           .help("input hdf5 file and dataset for holograms")
           .required().nargs(2);

    program.add_argument("--output_file", "-o")
           .help("output hdf5 file and dataset for registered holograms")
           .required().nargs(2);
           
    program.add_argument("--batch_size", "-b")
           .help("batch size of angles processed at a time")
           .required().scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Read holograms and image size from user inputs
    std::vector<hsize_t> dims;
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-i");
    IOUtils::readDataDims(inputs[0], inputs[1], dims);
    if (dims.size() != 4) {
        throw std::runtime_error("Invalid holograms or dimensions!");
    }

    int totalAngles = static_cast<int>(dims[0]);
    int numHolograms = static_cast<int>(dims[1]);
    int rows = static_cast<int>(dims[2]);
    int cols = static_cast<int>(dims[3]);

    int batchSize = program.get<int>("-b");
    FArray holograms(batchSize * numHolograms * rows * cols);

    std::vector<std::string> outputs = program.get<std::vector<std::string>>("-o");
    if (inputs[0] == outputs[0]) {
        throw std::runtime_error("Input and output files cannot be the same!");
    }
    IOUtils::createFileDataset(outputs[0], outputs[1], dims);

    std::vector<itk::simple::Image> images(numHolograms);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < totalAngles / batchSize; i++) {
       std::cout << "Processing batch " << i << std::endl;
       IOUtils::read4DimData(inputs[0], inputs[1], holograms, i * batchSize, batchSize);

       for (int j = 0; j < batchSize; j++) {
          ImageUtils::convertVecToImgs(holograms.data() + j * numHolograms * rows * cols, images, rows, cols);
          for (int k = 1; k < numHolograms; k++) {
              ImageUtils::registerImage(images[0], images[k]);
          }
          ImageUtils::convertImgsToVec(images, holograms.data() + j * numHolograms * rows * cols, rows, cols);
       }

       IOUtils::write4DimData(outputs[0], outputs[1], holograms, dims, i * batchSize);
    }
        
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}