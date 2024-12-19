#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "../include/holo_recons.h"
#include "../include/imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_recons_iter");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_files", "-I")
           .help("input hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one).scan<'g', float>();

    program.add_argument("--iterations", "-i")
           .help("the number of iterations")
           .required().scan<'i', int>().default_value(200);

    program.add_argument("--guess_phase_files", "-g")
           .help("hdf5 file and dataset of initial guess phase")
           .nargs(2);

    program.add_argument("--algorithm", "-a")
           .help("phase retrieval algorithm [0: ap, 1: raar, 2: hio, 3: drap]")
           .required().default_value(0).scan<'i', int>();

    program.add_argument("--algorithm_parameters", "-P")
           .help("parameters corresponding to different algorithm [default for hio and drap: 0.7]\n"
                 "default for raar: 0.75, 0.99, 20")
           .nargs(1, 3).scan<'g', float>();

    program.add_argument("--padding_size", "-s")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();
    
    program.add_argument("--amplitude_limits", "-l")
           .help("minimum and maximum amplitude constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--projection_type", "-t")
           .help("projection computing type [0: averaged, 1: sequential, 2: cyclic]")
           .required().default_value(0).scan<'i', int>();

    program.add_argument("--kernel_method", "-m")
           .help("propagation kernel method [0: fourier, 1: chirp, 2: chirplimited]")
           .required().default_value(0).scan<'i', int>();
    
    program.add_argument("--error_calculation", "-e")
           .help("whether to calculate iteration error")
           .default_value(false).implicit_value(true);

    program.add_argument("--padding_type", "-p")
           .help("type of padding matrix around [0: constant, 1: replicate, 2: fadeout]")
           .default_value(1).scan<'i', int>();

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
    std::vector<std::string> inputs = program.get<std::vector<std::string>>("-I");
    IOUtils::readProcessedGrams(inputs[0], inputs[1], holograms, dims);
    if (holograms.empty() || dims.empty() || dims.size() != 3) {
       throw std::runtime_error("Invalid holograms or dimensions!");
    }
    
    int numHolograms = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    IntArray imSize {rows, cols};

    auto fresnel_input = program.get<FArray>("-f");
    F2DArray fresnelNumbers;
    for (const auto &group: fresnel_input) {
        fresnelNumbers.push_back({group});
    }

    auto iterations = program.get<int>("-i");

    // Read initial phase and image size from user inputs
    FArray initialPhase;
    if (program.is_used("-g")) {
       std::vector<std::string> input_phase = program.get<std::vector<std::string>>("-g");
       IOUtils::readPhasegrams(input_phase[0], input_phase[1], initialPhase, dims);
    }

    auto algorithm = static_cast<ProjectionSolver::Algorithm>(program.get<int>("-a"));

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

    if (program.is_used("-s")) {
        padSize = program.get<IntArray>("-s");
        padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
    }

    FArray ampLimits {0, FloatInf};
    if (program.is_used("-l")) {
       ampLimits = program.get<FArray>("-l");
    }

    auto projectionType = static_cast<PMagnitudeCons::Type>(program.get<int>("-t"));
    auto kernelMethod = static_cast<CUDAPropKernel::Type>(program.get<int>("-m"));
    bool calcError = program.get<bool>("-e") ? true : false;

    auto start = std::chrono::high_resolution_clock::now();
    
    F2DArray result = PhaseRetrieval::reconstruct_iter(holograms, numHolograms, imSize, fresnelNumbers, iterations, initialPhase, algorithm,
                                     parameters, padSize, ampLimits[0], ampLimits[1], projectionType, kernelMethod, padType, calcError);
        
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    // Display the 2 results and save to HDF5 file
    // OUtils::savePhasegrams("/home/hug/Downloads/HoloTomo_Data/reconsfile.h5", "phasedata", result[0], imSize[0], imSize[1]);
    // ImageUtils::displayNDArray(result, imSize[0], imSize[1], std::vector<std::string>{"phase", "amplitude"});

    return 0;
}