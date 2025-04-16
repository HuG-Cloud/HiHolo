#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "holo_recons.h"
#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_recons_angles");
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

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one).scan<'g', float>();

    program.add_argument("--iterations", "-i")
           .help("the number of iterations")
           .required().scan<'i', int>().default_value(200);

    program.add_argument("--algorithm", "-a")
           .help("phase retrieval algorithm [0: ap, 1: raar, 2: hio, 3: drap]")
           .required().default_value(0).scan<'i', int>();

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
    
    program.add_argument("--support_size", "-s")
           .help("size of support")
           .nargs(2).scan<'i', int>();

    program.add_argument("--support_outside_value", "-sv")
           .help("value outside support constraint region")
           .default_value(1.0f).scan<'g', float>();
    
    program.add_argument("--amplitude_limits", "-al")
           .help("minimum and maximum amplitude constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--projection_type", "-t")
           .help("projection computing type [0: averaged, 1: sequential, 2: cyclic]")
           .required().default_value(0).scan<'i', int>();

    program.add_argument("--kernel_method", "-m")
           .help("propagation kernel method [0: fourier, 1: chirp, 2: chirplimited]")
           .required().default_value(0).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Read dimensions of holograms
    std::vector<hsize_t> dims;
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-I");
    IOUtils::readDataDims(inputs[0], inputs[1], dims);
    if (dims.size() != 4) {
        throw std::runtime_error("Invalid holograms or dimensions!");
    }

    int totalAngles = static_cast<int>(dims[0]);
    int numHolograms = static_cast<int>(dims[1]);
    int rows = static_cast<int>(dims[2]);
    int cols = static_cast<int>(dims[3]);
    IntArray imSize {rows, cols};

    int batchSize = program.get<int>("-b");
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

    std::vector<std::string> outputs = program.get<std::vector<std::string>>("-O");
    std::vector<hsize_t> outputDims = {dims[0], dims[2], dims[3]};
    IOUtils::createFileDataset(outputs[0], outputs[1], outputDims);
    
    auto reconstructor = PhaseRetrieval::Reconstructor(batchSize, numHolograms, imSize, fresnelNumbers, iterations, algorithm,
                                                       parameters, phaseLimits[0], phaseLimits[1], ampLimits[0], ampLimits[1],
                                                       support, outsideValue, padSize, padType, padValue, projectionType, kernelMethod);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < totalAngles / batchSize; i++) {
       std::cout << "Processing batch " << i << std::endl;
       IOUtils::read4DimData(inputs[0], inputs[1], holograms, i * batchSize, batchSize);
       if (!initialPhase.empty()) {
           IOUtils::read3DimData(inputPhase[0], inputPhase[1], initialPhase, i * batchSize, batchSize);
       }
       auto result = reconstructor.reconsBatch(holograms, initialPhase);
       IOUtils::write3DimData(outputs[0], outputs[1], result, outputDims, i * batchSize);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Finished phase retrieval for " << totalAngles << " angles!" << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;

    return 0;
}