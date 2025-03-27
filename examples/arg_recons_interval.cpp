#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include "holo_recons.h"
#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_recons_interval");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_file", "-I")
           .help("input hdf5 file and dataset for holograms")
           .required().nargs(2);

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one)
           .scan<'g', float>();

    program.add_argument("--iterations", "-i")
           .help("the number of iterations")
           .required().scan<'i', int>().default_value(200);

    program.add_argument("--guess_phase_file", "-g")
           .help("hdf5 file and dataset of initial guess phase")
           .nargs(2);

    program.add_argument("--algorithm", "-a")
           .help("phase retrieval algorithm [0:ap, 1:raar, 2:hio, 3:drap, 4:apwp]")
           .required().default_value(0).scan<'i', int>();

    program.add_argument("--plot_interval", "-pi")
           .help("plot intervals for reconstruction")
           .required().default_value(10).scan<'i', int>();

    program.add_argument("--algorithm_parameters", "-P")
           .help("parameters corresponding to different algorithm [default for hio and drap: 0.7]\n"
                 "default for raar: 0.75, 0.99, 20")
           .nargs(1, 3).scan<'g', float>();

    program.add_argument("--amplitude_limits", "-al")
           .help("minimum and maximum amplitude constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--phase_limits", "-pl")
           .help("minimum and maximum phase constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--padding_size", "-s")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();

    program.add_argument("--input_probe_file", "-ip")
           .help("input hdf5 file and dataset for probes")
           .nargs(2);

    program.add_argument("--guess_probe_phase", "-gp")
           .help("hdf5 file and dataset of initial guess phase for probe")
           .nargs(2);

    program.add_argument("--projection_type", "-t")
           .help("projection computing type [0:averaged, 1:sequential, 2:cyclic]")
           .required().default_value(0).scan<'i', int>();

    program.add_argument("--kernel_method", "-m")
           .help("propagation kernel method [0:fourier, 1:chirp, 2:chirplimited]")
           .required().default_value(0).scan<'i', int>();
    
    program.add_argument("--error_calculation", "-e")
           .help("whether to calculate iteration error")
           .default_value(false).implicit_value(true);

    program.add_argument("--padding_type", "-p")
           .help("type of padding matrix around [0:constant, 1:replicate, 2:fadeout]")
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
    if (holograms.empty() || dims.size() != 3) {
       throw std::runtime_error("Invalid holograms or dimensions!");
    }
    
    int numHolograms = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    IntArray imSize {rows, cols};

    // Process fresnel numbers
    auto fresnel_input = program.get<FArray>("-f");
    F2DArray fresnelNumbers;
    for (const auto &group: fresnel_input) {
        fresnelNumbers.push_back({group});
    }

    auto iterations = program.get<int>("-i");
    auto plotInterval = program.get<int>("-pi");

    // Read initial phase and image size from user inputs
    FArray initialPhase;
    std::vector<std::string> input_phase;
    if (program.is_used("-g")) {
       input_phase = program.get<std::vector<std::string>>("-g");
       IOUtils::readPhasegrams(input_phase[0], input_phase[1], initialPhase, dims);
       if (initialPhase.empty() || dims.size() != 2) {
            throw std::runtime_error("Invalid initial phase or dimensions!");
       }
    }

    // Get and print iterative algorithm
    auto algorithm = static_cast<ProjectionSolver::Algorithm>(program.get<int>("-a"));
    std::cout << "Choosing algorithm: ";
    switch (algorithm) {
       case ProjectionSolver::AP: std::cout << "AP"; break;
       case ProjectionSolver::RAAR: std::cout << "RAAR"; break;
       case ProjectionSolver::HIO: std::cout << "HIO"; break;
       case ProjectionSolver::DRAP: std::cout << "DRAP"; break;
       case ProjectionSolver::APWP: std::cout << "AP with Probe"; break;
       default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

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

    FArray ampLimits {0, FloatInf};
    if (program.is_used("-al")) {
       ampLimits = program.get<FArray>("-al");
    }

    FArray phaLimits {-FloatInf, FloatInf};
    if (program.is_used("-pl")) {
       phaLimits = program.get<FArray>("-pl");
    }

    FArray support;
    float outSideValue = 1.0f;

    // Process padding parameters
    IntArray padSize;
    CUDAUtils::PaddingType padType;
    if (program.is_used("-s")) {
        padSize = program.get<IntArray>("-s");
        padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
    }

    // Process probe field parameters
    FArray probeGrams, initProbePhase;
    if (algorithm == ProjectionSolver::Algorithm::APWP) {
       if (!program.is_used("-ip")) {
           throw std::runtime_error("Probe field is required for APWP algorithm!");
       }

       inputs = program.get<std::vector<std::string>>("-ip");
       IOUtils::readProcessedGrams(inputs[0], inputs[1], probeGrams, dims);
       if (probeGrams.empty() || dims.size() != 3) {
           throw std::runtime_error("Invalid probe grams or dimensions!");
       }

       if (program.is_used("-gp")) {
           input_phase = program.get<std::vector<std::string>>("-gp");
           IOUtils::readPhasegrams(input_phase[0], input_phase[1], initProbePhase, dims);
           if (initProbePhase.empty() || dims.size() != 2) {
                throw std::runtime_error("Invalid initial probe phase or dimensions!");
           }
       }
    }

    // Get and print projection method    
    auto projectionType = static_cast<PMagnitudeCons::Type>(program.get<int>("-t"));
    std::cout << "Choosing projection method: ";
    switch (projectionType) {
       case PMagnitudeCons::Averaged: std::cout << "Averaged"; break;
       case PMagnitudeCons::Sequential: std::cout << "Sequential"; break;
       case PMagnitudeCons::Cyclic: std::cout << "Cyclic"; break;
       default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    // Get and print kernel method
    auto kernelMethod = static_cast<CUDAPropKernel::Type>(program.get<int>("-m"));
    std::cout << "Choosing propagation kernel: ";
    switch (kernelMethod) {
       case CUDAPropKernel::Fourier: std::cout << "Fourier"; break;
       case CUDAPropKernel::Chirp: std::cout << "Chirp"; break;
       case CUDAPropKernel::ChirpLimited: std::cout << "Chirp Limited"; break;
       default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    F2DArray result, residuals;
    bool calcError = program.get<bool>("-e");
    if (calcError) {
        residuals = F2DArray(2, FArray(iterations));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations / plotInterval; ++i) {
        result = PhaseRetrieval::reconstruct_iter(holograms, numHolograms, imSize, fresnelNumbers, plotInterval, initialPhase,
                                                  algorithm, parameters, padSize, phaLimits[0], phaLimits[1], ampLimits[0], 
                                                  ampLimits[1], support, outSideValue, projectionType, kernelMethod, padType,
                                                  probeGrams, initProbePhase, calcError);
    
        initialPhase = result[0];
        if (algorithm == ProjectionSolver::APWP) {
            initProbePhase = result[2];
        }

        if (calcError) {
            std::copy(result[3].begin(), result[3].end(), residuals[0].begin() + i * plotInterval);
            std::copy(result[4].begin(), result[4].end(), residuals[1].begin() + i * plotInterval);
        }

        ImageUtils::displayPhase(result[0], imSize[0], imSize[1], "phase reconstructed by " + \
                                 std::to_string((i + 1) * plotInterval) + " iterations");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}