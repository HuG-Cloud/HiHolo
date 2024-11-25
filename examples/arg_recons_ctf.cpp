#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>

#include "../include/holo_recons.h"
#include "../include/imageio_utils.h"

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("recons_gpu");
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
           .scan<'g', float>();

    program.add_argument("--high_freq_lim", "-H")
           .help("regularisation parameters for high frequencies [default: 1e-1]")
           .scan<'g', float>();

    program.add_argument("--padding_size", "-s")
           .help("size to pad on holograms")
           .nargs(2).scan<'i', int>();

    program.add_argument("--padding_type", "-p")
           .help("type of padding matrix around [0: const, 1: replicate, 2: fadeout]")
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

    float ratio = program.get<float>("-r");

    IntArray padSize;
    CUDAUtils::PaddingType padType;

    if (program.is_used("-s")) {
        padSize = program.get<IntArray>("-s");
        padType = static_cast<CUDAUtils::PaddingType>(program.get<int>("-p"));
    }

    // Read regularisation parameters
    float lowFreqLim = 1e-3f;
    float highFreqLim = 1e-1f;
    if (program.is_used("-L")) {
       lowFreqLim = program.get<float>("-L");
    }
    if (program.is_used("-H")) {
       highFreqLim = program.get<float>("-H");
    }
    
    // auto start = std::chrono::high_resolution_clock::now();
    
    FArray phase = PhaseRetrieval::reconstruct_ctf(holograms, numHolograms, imSize, fresnelNumbers, lowFreqLim, highFreqLim, ratio, padSize, padType);
    F2DArray result {phase};
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

    // Display the phase result and save to HDF5 file
    // OUtils::savePhasegrams("/home/hug/Downloads/HoloTomo_Data/reconsfile.h5", "phasedata", result[0], imSize[0], imSize[1]);
    ImageUtils::displayNDArray(result, imSize[0], imSize[1], std::vector<std::string>{"phase"});

    return 0;
}