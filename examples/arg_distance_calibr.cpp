#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("distance_calibration");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-i")
           .help("input hdf5 file and dataset for periodic holograms")
           .required().nargs(2);

    program.add_argument("--period_length", "-l")
           .help("period length of periodic object")
           .required().scan<'g', double>();
    
    program.add_argument("--pixel_size", "-p")
           .help("physical pixel size")
           .required().scan<'g', double>();

    program.add_argument("--step_size", "-s")
           .help("fixed step size for placement")
           .required().scan<'g', double>(); 

    program.add_argument("--num_steps", "-n")
           .help("number of steps for placement")
           .required().nargs(argparse::nargs_pattern::at_least_one)
           .scan<'g', double>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Read holograms and image size from user inputs
    FArray holograms;
    std::vector<hsize_t> dims;
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-i");
    IOUtils::readProcessedGrams(inputs[0], inputs[1], holograms, dims);
    if (holograms.empty() || dims.size() != 3) {
       throw std::runtime_error("Invalid holograms or dimensions!");
    }
    
    int numImages = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);

    double periodLength = program.get<double>("-l");
    double pixelSize = program.get<double>("-p");
    double stepSize = program.get<double>("-s");
    DArray numSteps = program.get<DArray>("-n");
    if (numSteps.size() != numImages) {
        throw std::runtime_error("Number of steps must be equal to number of images!");
    }

    auto start = std::chrono::high_resolution_clock::now();
    D2DArray parameters = ImageUtils::calibrateDistance(holograms, numImages, rows, cols,
                                                       periodLength, pixelSize, numSteps, stepSize);
    auto end = std::chrono::high_resolution_clock::now();

    // std::cout << "Distance calibrated: z1:" << parameters[2][0] << " z2:" << parameters[2][1] << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}