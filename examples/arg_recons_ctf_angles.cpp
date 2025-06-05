#include <argparse/argparse.hpp>
#include <chrono>

#include "holo_recons.h"
#include "io_utils.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    argparse::ArgumentParser program("holo_recons_ctf_angles");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_files", "-I")
           .help("input hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--output_files", "-O")
           .help("output hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--batch_size", "-b")
           .help("batch size of holograms processed at a time")
           .required().scan<'i', int>();

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one)
           .scan<'g', float>();

    program.add_argument("--device_numbers", "-d")
           .help("number of GPUs to use [default: all GPU resources]")
           .scan<'i', int>();

    program.add_argument("--ratio", "-r")
           .help("fixed ratio between absorption and phase shifts")
           .default_value(0.0f).scan<'g', float>();
    
    program.add_argument("--low_freq_lim", "-L")
           .help("regularisation parameters for low frequencies [default: 1e-3]")
           .default_value(1e-3f).scan<'g', float>();

    program.add_argument("--high_freq_lim", "-H")
           .help("regularisation parameters for high frequencies [default: 1e-1]")
           .default_value(1e-1f).scan<'g', float>();

    program.add_argument("--padding_size", "-S")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();

    program.add_argument("--padding_type", "-p")
           .help("type of padding matrix around [0: constant, 1: replicate, 2: fadeout]")
           .default_value(1).scan<'i', int>();

    program.add_argument("--padding_value", "-V")
           .help("value to pad on holograms and initial phase")
           .default_value(0.0f).scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        MPI_Finalize();
        return 1;
    }

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        throw std::runtime_error("No CUDA capable GPU device found!");
    }

    int devices = deviceCount;
    if (program.is_used("-d")) {
        devices = program.get<int>("-d");
        if (devices > deviceCount || devices <= 0) {
            throw std::runtime_error("Invalid number of GPUs to use!");
        }
    }

    int deviceId = rank % devices;
    cudaSetDevice(deviceId);

    // Read holograms and image size from user inputs
    std::vector<hsize_t> dims;
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-I");
    IOUtils::readDataDims(inputs[0], inputs[1], dims, MPI_COMM_WORLD);
    if (dims.size() != 4) {
        std::cerr << "Error: Input data must have 4 dimensions" << std::endl;
        return 1;
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
    FArray holograms(batchSize * numHolograms * rows * cols);

    auto fresnel_input = program.get<FArray>("-f");
    F2DArray fresnelNumbers;
    for (const auto &group: fresnel_input) {
        fresnelNumbers.push_back({group});
    }

    float ratio = program.get<float>("-r");

    IntArray padSize;
    CUDAUtils::PaddingType padType;
    float padValue;
    if (program.is_used("-S")) {
       padSize = program.get<IntArray>("-S");
       padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
       padValue = program.get<float>("-V");
    }

    // Read regularisation parameters
    float lowFreqLim = program.get<float>("-L");
    float highFreqLim = program.get<float>("-H");

    std::vector<std::string> outputs = program.get<std::vector<std::string>>("-O");
    if (outputs[0] == inputs[0]) {
       throw std::runtime_error("Input and output files cannot be the same!");
    }
    std::vector<hsize_t> outputDims {dims[0], dims[2], dims[3]};

    auto reconstructor = new PhaseRetrieval::CTFReconstructor(batchSize, numHolograms, imSize, fresnelNumbers,
                                                              lowFreqLim, highFreqLim, ratio, padSize, padType, padValue);
    
    // Create output dataset before processing
    if(!IOUtils::createFileDataset(outputs[0], outputs[1], outputDims, MPI_COMM_WORLD)) {
        throw std::runtime_error("Failed to create output file or dataset!");
    }  
    auto start = std::chrono::high_resolution_clock::now();    

    for (int i = 0; i < numAngles / batchSize; i++) {
       if (rank == 0) {
           std::cout << "Processing batch " << i + 1 << "/" << numAngles / batchSize << std::endl;
       }
       int globalIndex = startAngle + i * batchSize;
       IOUtils::read4DimData(inputs[0], inputs[1], holograms, globalIndex, batchSize, MPI_COMM_WORLD);
       auto result = reconstructor->reconsBatch(holograms);
       IOUtils::write3DimData(outputs[0], outputs[1], result, outputDims, globalIndex, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "Finished CTF reconstruction for " << totalAngles << " angles on " << devices << " GPUs!" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;
    }

    delete reconstructor;
    MPI_Finalize();

    return 0;
}