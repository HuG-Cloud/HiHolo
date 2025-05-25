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

    argparse::ArgumentParser program("holo_recons_ite_angles");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_files", "-I")
           .help("input hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--output_files", "-O")
           .help("output hdf5 file and dataset")
           .required().nargs(2);
    
    program.add_argument("--batch_size", "-b")
           .help("batch size of angles processed at a time")
           .required().scan<'i', int>();

    program.add_argument("--device_numbers", "-d")
           .help("number of GPUs to use [default: all GPU resources]")
           .scan<'i', int>();

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one)
           .scan<'g', float>();

    program.add_argument("--iterations", "-i")
           .help("the number of iterations")
           .default_value(200).scan<'i', int>();

    program.add_argument("--algorithm", "-a")
           .help("phase retrieval algorithm [0: ap, 1: raar, 2: hio, 3: drap]")
           .default_value(0).scan<'i', int>();

    program.add_argument("--algorithm_parameters", "-P")
           .help("parameters corresponding to different algorithm [default for hio and drap: 0.7]\n"
                 "default for raar: 0.75, 0.99, 20")
           .nargs(1, 3).scan<'g', float>();
    
    program.add_argument("--guess_phase_file", "-g")
           .help("hdf5 file and dataset of initial phase guess")
           .nargs(2);

    program.add_argument("--padding_size", "-S")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();

    program.add_argument("--padding_type", "-p")
           .help("type of padding matrix around [0: constant, 1: replicate, 2: fadeout]")
           .default_value(1).scan<'i', int>();

    program.add_argument("--padding_value", "-V")
           .help("value to pad on holograms and initial phase")
           .default_value(0.0f).scan<'g', float>();

    program.add_argument("--phase_limits", "-pl")
           .help("minimum and maximum phase constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--amplitude_limits", "-al")
           .help("minimum and maximum amplitude constraints")
           .nargs(2).scan<'g', float>();    

    program.add_argument("--support_size", "-s")
           .help("size of support")
           .nargs(2).scan<'i', int>();

    program.add_argument("--support_outside_value", "-sv")
           .help("value outside support constraint region")
           .default_value(1.0f).scan<'g', float>();

    program.add_argument("--projection_type", "-t")
           .help("projection computing type [0: averaged, 1: sequential, 2: cyclic]")
           .default_value(0).scan<'i', int>();

    program.add_argument("--kernel_method", "-m")
           .help("propagation kernel method [0: fourier, 1: chirp, 2: chirplimited]")
           .default_value(0).scan<'i', int>();

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

    // Read dimensions of holograms
    std::vector<hsize_t> dims;
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-I");
    IOUtils::readDataDims(inputs[0], inputs[1], dims, MPI_COMM_WORLD);
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
    FArray holograms(batchSize * numHolograms * rows * cols);

    auto fresnel_input = program.get<FArray>("-f");
    F2DArray fresnelNumbers;
    for (const auto &group: fresnel_input) {
        fresnelNumbers.push_back({group});
    }

    auto iterations = program.get<int>("-i");
    auto algorithm = static_cast<ProjectionSolver::Algorithm>(program.get<int>("-a"));

    FArray initialPhase;
    std::vector<std::string> inputPhase;
    if (program.is_used("-g")) {
       inputPhase = program.get<std::vector<std::string>>("-g");
       initialPhase.resize(batchSize * rows * cols);
    }

    // Read algorithm parameters
    FArray parameters;
    if (program.is_used("-P")) {
       parameters = program.get<FArray>("-P");
    } else if (algorithm == ProjectionSolver::Algorithm::RAAR) {
       parameters = {0.75, 0.99, 20};
    } else if (algorithm == ProjectionSolver::Algorithm::HIO || 
               algorithm == ProjectionSolver::Algorithm::DRAP) {
       parameters = {0.7};
    }

    IntArray padSize;
    CUDAUtils::PaddingType padType;
    float padValue;
    if (program.is_used("-S")) {
        padSize = program.get<IntArray>("-S");
        padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
        padValue = program.get<float>("-V");
    }

    FArray ampLimits {0, FloatInf};
    if (program.is_used("-al")) {
       ampLimits = program.get<FArray>("-al");
    }

    FArray phaseLimits {-FloatInf, FloatInf};
    if (program.is_used("-pl")) {
       phaseLimits = program.get<FArray>("-pl");
    }

    IntArray support;
    float outsideValue;
    if (program.is_used("-s")) {
       support = program.get<IntArray>("-s");
       outsideValue = program.get<float>("-sv");
    }

    auto projectionType = static_cast<PMagnitudeCons::Type>(program.get<int>("-t"));
    auto kernelMethod = static_cast<CUDAPropKernel::Type>(program.get<int>("-m"));

    if (rank == 0) {
        std::cout << "Choosing algorithm: ";
        switch (algorithm) {
            case ProjectionSolver::AP: std::cout << "AP"; break;
            case ProjectionSolver::RAAR: std::cout << "RAAR"; break;
            case ProjectionSolver::HIO: std::cout << "HIO"; break;
            case ProjectionSolver::DRAP: std::cout << "DRAP"; break;
            default: std::cout << "Unknown!";
        }

        std::cout << std::endl << "Choosing projection method: ";
        switch (projectionType) {
            case PMagnitudeCons::Averaged: std::cout << "Averaged"; break;
            case PMagnitudeCons::Sequential: std::cout << "Sequential"; break;
            case PMagnitudeCons::Cyclic: std::cout << "Cyclic"; break;
            default: std::cout << "Unknown!";
        }
        
        std::cout << std::endl << "Choosing propagation kernel type: ";
        switch (kernelMethod) {
            case CUDAPropKernel::Fourier: std::cout << "Fourier"; break;
            case CUDAPropKernel::Chirp: std::cout << "Chirp"; break;
            case CUDAPropKernel::ChirpLimited: std::cout << "ChirpLimited"; break;
            default: std::cout << "Unknown!";
        }
        std::cout << std::endl;
    }

    std::vector<std::string> outputs = program.get<std::vector<std::string>>("-O");
    if (outputs[0] == inputs[0]) {
        throw std::runtime_error("Input and output files cannot be the same!");
    }
    std::vector<hsize_t> outputDims = {dims[0], dims[2], dims[3]};
    
    auto reconstructor = PhaseRetrieval::Reconstructor(batchSize, numHolograms, imSize, fresnelNumbers, iterations, algorithm,
                                                       parameters, phaseLimits[0], phaseLimits[1], ampLimits[0], ampLimits[1],
                                                       support, outsideValue, padSize, padType, padValue, projectionType, kernelMethod);
    
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
       if (!initialPhase.empty()) {
           IOUtils::read3DimData(inputPhase[0], inputPhase[1], initialPhase, globalIndex, batchSize, MPI_COMM_WORLD);
       }
       auto result = reconstructor.reconsBatch(holograms, initialPhase);
       IOUtils::write3DimData(outputs[0], outputs[1], result, outputDims, globalIndex, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "Finished phase retrieval for " << totalAngles << " angles on " << devices << " GPUs" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}