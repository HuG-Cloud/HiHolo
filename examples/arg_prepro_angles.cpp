#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "holo_recons.h"

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    argparse::ArgumentParser program("data_prepro_angles");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-i")
           .help("input hdf5 file of raw detector data")
           .required();

    program.add_argument("--output_file", "-o")
           .help("output hdf5 file of preprocessed data")
           .required();
           
    program.add_argument("--batch_size", "-b")
           .help("batch size of angles processed at a time")
           .required().scan<'i', int>();

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
        MPI_Finalize();
        return 1;
    }

    std::vector<hsize_t> dims;
    std::string input = program.get<std::string>("-i");
    U16Array dark, flat;
    IOUtils::readSingleGram(input, "dark", dark, dims, MPI_COMM_WORLD);
    IOUtils::readSingleGram(input, "flat", flat, dims, MPI_COMM_WORLD);
    IOUtils::readDataDims(input, "data", dims, MPI_COMM_WORLD);
    if (dims.size() != 4) {
        throw std::runtime_error("Invalid holograms or dimensions!");
    }

    int totalAngles = static_cast<int>(dims[0]);
    int numHolograms = static_cast<int>(dims[1]);
    int rows = static_cast<int>(dims[2]);
    int cols = static_cast<int>(dims[3]);
    IntArray imSize {rows, cols};
    
    // Each process handles the same number of angles
    int numAngles = totalAngles / size;
    int startAngle = rank * numAngles;
    int batchSize = program.get<int>("-b");
    batchSize = std::min(batchSize, numAngles);
    if (numAngles % batchSize != 0) {
        throw std::runtime_error("Number of angles must be divisible by batch size!");
    }
    U16Array rawData(batchSize * numHolograms * rows * cols);

    int kernelSize = program.get<int>("-k");
    float threshold = program.get<float>("-t");
    int rangeRows = program.get<int>("-r");
    int rangeCols = program.get<int>("-c");
    int movmeanSize = program.get<int>("-m");
    std::string method = program.get<std::string>("-M");

    std::string output = program.get<std::string>("-o");
    if (input == output) {
        throw std::runtime_error("Input and output file cannot be the same!");
    }

    auto preprocessor = PhaseRetrieval::Preprocessor(batchSize, numHolograms, imSize, dark, flat,
                                                     kernelSize, threshold, rangeRows, rangeCols,
                                                     movmeanSize, method);
    
    if(!IOUtils::createFileDataset(output, "holodata", dims, MPI_COMM_WORLD)) {
        throw std::runtime_error("Failed to create output file or dataset!");
    }
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numAngles / batchSize; i++) {
       if (rank == 0) {
          std::cout << "Processing batch " << i + 1 << "/" << numAngles / batchSize << std::endl;
       }
       int globalIndex = startAngle + i * batchSize;
       IOUtils::read4DimData(input, "data", rawData, globalIndex, batchSize, MPI_COMM_WORLD);
       auto holograms = preprocessor.processBatch(rawData);
       IOUtils::write4DimData(output, "holodata", holograms, dims, globalIndex, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        std::cout << "Finished data preprocessing for " << totalAngles << " angles!" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}