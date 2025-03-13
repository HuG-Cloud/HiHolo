#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include "holo_recons.h"
#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_recons_pipeline");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_files", "-I")
           .help("input hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--fresnel_numbers", "-f")
           .help("list of fresnel numbers corresponding to holograms")
           .required().nargs(argparse::nargs_pattern::at_least_one).scan<'g', float>();

    program.add_argument("--ratio", "-r")
           .help("fixed ratio between absorption and phase shifts")
           .required().scan<'g', float>().default_value(0.0f);

    program.add_argument("--low_freq_lim", "-L")
           .help("regularisation parameters for low frequencies [default: 1e-3]")
           .required().scan<'g', float>().default_value(1e-3f);

    program.add_argument("--high_freq_lim", "-H")
           .help("regularisation parameters for high frequencies [default: 1e-1]")
           .required().scan<'g', float>().default_value(1e-1f);

    program.add_argument("--algorithm", "-a")
           .help("iterative phase retrieval algorithm sequence [0: ap, 1: raar, 2: hio]")
           .required().nargs(3).scan<'i', int>();
    
    program.add_argument("--iterations", "-i")
           .help("the number of iterations for each algorithm")
           .required().nargs(3).scan<'i', int>();

    program.add_argument("--padding_size", "-s")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();
    
    program.add_argument("--amplitude_limits", "-l")
           .help("minimum and maximum amplitude constraints")
           .nargs(2).scan<'g', float>();

    program.add_argument("--kernel_method", "-m")
           .help("propagation kernel method [0: fourier, 1: chirp, 2: chirplimited]")
           .required().default_value(0).scan<'i', int>();

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
    
    int numHolograms = static_cast<int>(dims[0]);
    int rows = static_cast<int>(dims[1]);
    int cols = static_cast<int>(dims[2]);
    IntArray imSize {rows, cols};

    auto fresnel_input = program.get<FArray>("-f");
    F2DArray fresnelNumbers;
    for (const auto &group: fresnel_input) {
        fresnelNumbers.push_back({group});
    }

    auto algorithms = program.get<IntArray>("-a");
    auto iterations = program.get<IntArray>("-i");
    std::vector<ProjectionSolver::Algorithm> algorithm_sequence;
    for(const auto& alg: algorithms) {
        algorithm_sequence.push_back(static_cast<ProjectionSolver::Algorithm>(alg));
    }

    IntArray padSize;
    CUDAUtils::PaddingType padType;

    if (program.is_used("-s")) {
        padSize = program.get<IntArray>("-s");
        padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
    }

    FArray ampLimits {0.0f, FloatInf};
    if (program.is_used("-l")) {
       ampLimits = program.get<FArray>("-l");
    }

    float ratio = program.get<float>("-r");
    // Read regularisation parameters
    float lowFreqLim = program.get<float>("-L");
    float highFreqLim = program.get<float>("-H");

    auto projectionType = PMagnitudeCons::Type::Averaged;
    auto kernelMethod = static_cast<CUDAPropKernel::Type>(program.get<int>("-m"));

    FArray phase = PhaseRetrieval::reconstruct_ctf(holograms, numHolograms, imSize, fresnelNumbers, 
                                                    lowFreqLim, highFreqLim, ratio, padSize, padType);
    ImageUtils::displayPhase(phase, imSize[0], imSize[1], "Phase Reconstructed by CTF");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Iterative phase retrieval and display results
    FArray parameters;
    std::string alg_name;
    for(int i = 0; i < algorithm_sequence.size(); ++i) {
        switch (algorithm_sequence[i]) {
            case ProjectionSolver::Algorithm::RAAR:
                parameters = {0.75, 0.99, 20};
                alg_name = "RAAR";
                break;
            case ProjectionSolver::Algorithm::AP:
                parameters = {};
                alg_name = "AP";
                break;
            case ProjectionSolver::Algorithm::HIO:
                parameters = {0.7};
                alg_name = "HIO";
                break;
            case ProjectionSolver::Algorithm::DRAP:
                parameters = {0.7};
                alg_name = "DRAP";
                break;
        }
        F2DArray result = PhaseRetrieval::reconstruct_iter(holograms, numHolograms, imSize, fresnelNumbers, 
                                                          iterations[i], phase, algorithm_sequence[i],
                                                          parameters, padSize, ampLimits[0], ampLimits[1], 
                                                          projectionType, kernelMethod, padType);
        phase = result[0];
        ImageUtils::displayPhase(phase, imSize[0], imSize[1], "Phase Reconstructed by " + alg_name);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
     
    // IOUtils::savePhasegrams("/home/hug/Downloads/HoloTomo_Data/reconsfile.h5", "phasedata", result[0], imSize[0], imSize[1]);

    return 0;
}