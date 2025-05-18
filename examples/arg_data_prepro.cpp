#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "holo_recons.h"
#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("data_preprocessing");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-i")
           .help("input hdf5 file of raw detector data")
           .required();

    program.add_argument("--output_file", "-o")
           .help("output hdf5 file of preprocessed data")
           .required();
           
    program.add_argument("--is_apwp", "-a")
           .help("whether the phase are retrieved by apwp")
           .default_value(false).implicit_value(true);

    program.add_argument("--output_probe_file", "-p")
           .help("output hdf5 file of preprocessed probe data");

    program.add_argument("--kernel_size", "-k")
           .help("kernel size for removing outliers")
           .default_value(3).scan<'i', int>();

    program.add_argument("--threshold", "-t")
           .help("threshold for removing outliers")
           .default_value(2.0f).scan<'g', float>();

    program.add_argument("--range_rows", "-r")
           .help("range of rows to remove stripes")
           .default_value(0).scan<'i', int>();

    program.add_argument("--range_cols", "-c")
           .help("range of columns to remove stripes")
           .default_value(0).scan<'i', int>();

    program.add_argument("--movmean_size", "-m")
           .help("size of moving average for removing stripes")
           .default_value(5).scan<'i', int>();

    program.add_argument("--removal_method", "-M")
           .help("calculation method for removing stripes")
           .default_value("mul");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Read raw data and image size from user inputs
    U16Array rawData, dark, flat;
    std::vector<hsize_t> dims;
    std::string inputFile = program.get<std::string>("-i");
    std::vector<std::string> datasetNames {"data", "dark", "flat"};
    IOUtils::readRawData(inputFile, datasetNames, dims, rawData, dark, flat);

    int numImages = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    IntArray imSize {rows, cols};

    bool isAPWP = program.get<bool>("-a");
    if (isAPWP) {
       if (rawData.size() != flat.size()) {
           throw std::runtime_error("APWP requires the same number of probe and object images!");
       }
    }

    int kernelSize = program.get<int>("-k");
    float threshold = program.get<float>("-t");
    int rangeRows = program.get<int>("-r");
    int rangeCols = program.get<int>("-c");
    int movmeanSize = program.get<int>("-m");
    std::string method = program.get<std::string>("-M");

    std::string outputFile = program.get<std::string>("-o");
    if (inputFile == outputFile) {
        throw std::runtime_error("Input and output files cannot be the same!");
    }
    
    std::string outputProbeFile;
    if (isAPWP) {
        if (!program.is_used("-p")) {
            throw std::runtime_error("When APWP is enabled, --output_probe_file must be specified!");
        }
        outputProbeFile = program.get<std::string>("-p");
        if (inputFile == outputProbeFile || outputFile == outputProbeFile) {
            throw std::runtime_error("Output probe file cannot be the same as input or output file!");
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    F2DArray result = PhaseRetrieval::preprocess_data(rawData, dark, flat, numImages, imSize,
                                                      isAPWP, kernelSize, threshold, rangeRows,
                                                      rangeCols, movmeanSize, method);

    IOUtils::save3DGrams(outputFile, "holodata", result[0], numImages, rows, cols);
    if (isAPWP) {
        IOUtils::save3DGrams(outputProbeFile, "holodata", result[1], numImages, rows, cols);
    }    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}