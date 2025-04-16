#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "holo_recons.h"
#include "imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("holo_recons_ctf");
    program.set_usage_max_line_width(120);

    // Add arguments to ArgumentParser object
    program.add_argument("--input_files", "-I")
           .help("input hdf5 file and dataset")
           .required().nargs(2);

    program.add_argument("--output_files", "-O")
           .help("output hdf5 file and dataset")
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
       
    auto start = std::chrono::high_resolution_clock::now();    
    FArray phase = PhaseRetrieval::reconstruct_ctf(holograms, numHolograms, imSize, fresnelNumbers, lowFreqLim,
                                                   highFreqLim, ratio, padSize, padType, padValue);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    F2DArray result {phase};
    // Display the phase result and save to HDF5 file
    ImageUtils::displayPhase(result[0], imSize[0], imSize[1], "phase");
    IOUtils::savePhasegrams(outputs[0], outputs[1], result[0], imSize[0], imSize[1]);

    return 0;
}