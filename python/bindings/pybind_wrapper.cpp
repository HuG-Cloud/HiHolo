#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>

#include "holo_recons.h"

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    return cv::Mat(buf.shape[0], buf.shape[1], CV_32F, (float*)buf.ptr);
}

py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    return py::array_t<float>(
        {mat.rows, mat.cols},
        {mat.cols * sizeof(float), sizeof(float)},
        mat.ptr<float>()
    );
}

PYBIND11_MODULE(hiholo, m) {
    m.doc() = "Python binding for holographic reconstruction using CTF and iterative methods";

    py::enum_<CUDAUtils::PaddingType>(m, "PaddingType")
        .value("Constant", CUDAUtils::PaddingType::Constant)
        .value("Replicate", CUDAUtils::PaddingType::Replicate)
        .value("Fadeout", CUDAUtils::PaddingType::Fadeout);
    
    py::enum_<ProjectionSolver::Algorithm>(m, "Algorithm")
        .value("AP", ProjectionSolver::Algorithm::AP)
        .value("RAAR", ProjectionSolver::Algorithm::RAAR)
        .value("HIO", ProjectionSolver::Algorithm::HIO)
        .value("DRAP", ProjectionSolver::Algorithm::DRAP)
        .value("APWP", ProjectionSolver::Algorithm::APWP)
        .value("EPI", ProjectionSolver::Algorithm::EPI);

    py::enum_<PMagnitudeCons::Type>(m, "ProjectionType")
        .value("Averaged", PMagnitudeCons::Type::Averaged)
        .value("Sequential", PMagnitudeCons::Type::Sequential)
        .value("Cyclic", PMagnitudeCons::Type::Cyclic);

    py::enum_<CUDAPropKernel::Type>(m, "PropKernelType")
        .value("Fourier", CUDAPropKernel::Type::Fourier)
        .value("Chirp", CUDAPropKernel::Type::Chirp)
        .value("ChirpLimited", CUDAPropKernel::Type::ChirpLimited);

    // Bind removeOutliers function with numpy array conversion
    m.def("removeOutliers", [](py::array_t<float> image, int kernelSize, float threshold) {
          cv::Mat mat = numpy_to_mat(image);
          ImageUtils::removeOutliers(mat, kernelSize, threshold);
          return mat_to_numpy(mat);
    }, "Remove outliers from an image using median filtering",
          py::arg("image"),
          py::arg("kernelSize") = 5,
          py::arg("threshold") = 2.0f);

    // Bind removeStripes function with numpy array conversion
    m.def("removeStripes", [](py::array_t<float> image, int rangeRows, int rangeCols,
                              int windowSize, const std::string& method) {
          cv::Mat mat = numpy_to_mat(image);
          ImageUtils::removeStripes(mat, rangeRows, rangeCols, windowSize, method);
          return mat_to_numpy(mat);
    }, "Remove stripes from an image by linear interpolation",
          py::arg("image"),
          py::arg("rangeRows") = 0,
          py::arg("rangeCols") = 0,
          py::arg("windowSize") = 5,
          py::arg("method") = "mul");

    m.def("computePSDs", [](py::array_t<float> images_array, int direction) {
          py::buffer_info buf = images_array.request();
          
          // Parse dimensions from numpy array
          int numImages, rows, cols;
          if (buf.ndim == 3) {
              // Shape: (numImages, rows, cols)
              numImages = buf.shape[0];
              rows = buf.shape[1];
              cols = buf.shape[2];
          } else {
              throw std::runtime_error("Images array must be 3D (numImages, rows, cols)");
          }
          
          // Convert numpy array to vector of cv::Mat
          std::vector<cv::Mat> cvImages(numImages);
          float* data_ptr = static_cast<float*>(buf.ptr);
          for (int i = 0; i < numImages; i++) {
              cvImages[i] = cv::Mat(rows, cols, CV_32F, data_ptr + i * rows * cols);
          }
          
          std::vector<cv::Mat> profiles(numImages);
          std::vector<cv::Mat> frequencies(numImages);
          
          DArray maxPSDs = ImageUtils::computePSDs(cvImages, direction, profiles, frequencies);
          
          // Convert results using efficient memory operations
          py::list result_list;
          result_list.append(py::cast(maxPSDs));
          
          py::list frequencies_list;
          for (const auto& freq : frequencies) {
              // Create std::vector from cv::Mat data for efficient conversion
              DArray freq_vec(freq.ptr<float>(), freq.ptr<float>() + freq.total());
              frequencies_list.append(py::cast(freq_vec));
          }
          result_list.append(frequencies_list);

          py::list profiles_list;
          for (const auto& profile : profiles) {
              // Create std::vector from cv::Mat data for efficient conversion
              DArray profile_vec(profile.ptr<float>(), profile.ptr<float>() + profile.total());
              profiles_list.append(py::cast(profile_vec));
          }
          result_list.append(profiles_list);

          return result_list;
    }, "Compute PSDs of a group of images, returns [maxPSDs, profiles, frequencies]",
          py::arg("images"),
          py::arg("direction"));

    // Bind distance calibration function - simplified direct binding
    m.def("calibrateDistance", &ImageUtils::calibrateDistance,
          "Calibrate distance using maximum PSD values",
          py::arg("maxPSD"),
          py::arg("nz"),
          py::arg("length"),
          py::arg("pixelSize"),
          py::arg("stepSize"));
    
    // Bind CTF reconstruction function with numpy array auto-parsing
    m.def("reconstruct_ctf", [](py::array_t<float> holograms_array, const F2DArray& fresnelNumbers,
                                float lowFreqLim, float highFreqLim, float betaDeltaRatio,
                                const IntArray& padSize, CUDAUtils::PaddingType padType, float padValue) {
          py::buffer_info buf = holograms_array.request();
          
          // Parse dimensions from numpy array
          int numImages, rows, cols;
          if (buf.ndim == 3) {
              // Shape: (numImages, rows, cols)
              numImages = buf.shape[0];
              rows = buf.shape[1];
              cols = buf.shape[2];
          } else if (buf.ndim == 2) {
              // Shape: (rows, cols) - single image
              numImages = 1;
              rows = buf.shape[0];
              cols = buf.shape[1];
          } else {
              throw std::runtime_error("Holograms array must be 2D or 3D");
          }
          
          IntArray imSize {rows, cols};
          
          // Convert numpy array to FArray (flatten)
          FArray holograms;
          float* data_ptr = static_cast<float*>(buf.ptr);
          holograms.assign(data_ptr, data_ptr + buf.size);
          
                    // Call the original C++ function
          FArray result = PhaseRetrieval::reconstruct_ctf(holograms, numImages, imSize, fresnelNumbers, 
                                                          lowFreqLim, highFreqLim, betaDeltaRatio, padSize,
                                                          padType, padValue);
          
          // Convert result to 2D numpy array
          auto output = py::array_t<float>(rows * cols);
          py::buffer_info output_buf = output.request();
          float* output_ptr = static_cast<float*>(output_buf.ptr);
          
          // Copy data
          std::copy(result.begin(), result.end(), output_ptr);
          
          // Reshape to 2D
          output.resize({rows, cols});
          return output;
    }, "CTF phase retrieval with auto-parsing from numpy array",
          py::arg("holograms"),
          py::arg("fresnelNumbers"),
          py::arg("lowFreqLim") = 1e-3f,
          py::arg("highFreqLim") = 1e-1f,
          py::arg("betaDeltaRatio") = 0.0f, 
          py::arg("padSize") = IntArray(),
          py::arg("padType") = CUDAUtils::PaddingType::Replicate, 
          py::arg("padValue") = 0.0f);
    
    // Bind iterative reconstruction function with numpy array auto-parsing
    m.def("reconstruct_iter", [](py::array_t<float> holograms_array, const F2DArray& fresnelNumbers,
                                int iterations, py::array_t<float> initialPhase_array, 
                                ProjectionSolver::Algorithm algorithm, const FArray& algoParameters,
                                float minPhase, float maxPhase, float minAmplitude, float maxAmplitude,
                                const IntArray& support, float outsideValue, const IntArray& padSize,
                                CUDAUtils::PaddingType padType, float padValue, PMagnitudeCons::Type projectionType,
                                CUDAPropKernel::Type kernelType, py::array_t<float> holoProbes_array,
                                py::array_t<float> initProbePhase_array, bool calcError) {
          
          py::buffer_info holo_buf = holograms_array.request();
          
          // Parse dimensions from numpy array
          int numImages, rows, cols;
          if (holo_buf.ndim == 3) {
              // Shape: (numImages, rows, cols)
              numImages = holo_buf.shape[0];
              rows = holo_buf.shape[1];
              cols = holo_buf.shape[2];
          } else if (holo_buf.ndim == 2) {
              // Shape: (rows, cols) - single image
              numImages = 1;
              rows = holo_buf.shape[0];
              cols = holo_buf.shape[1];
          } else {
              throw std::runtime_error("Holograms array must be 2D or 3D");
          }
          
          IntArray imSize {rows, cols};
          
          // Convert holograms numpy array to FArray (flatten)
          FArray holograms;
          float* holo_data_ptr = static_cast<float*>(holo_buf.ptr);
          holograms.assign(holo_data_ptr, holo_data_ptr + holo_buf.size);
          
          // Convert initialPhase array if provided
          FArray initialPhase;
          if (initialPhase_array.size() > 0) {
              py::buffer_info phase_buf = initialPhase_array.request();
              float* phase_data_ptr = static_cast<float*>(phase_buf.ptr);
              initialPhase.assign(phase_data_ptr, phase_data_ptr + phase_buf.size);
          }
          
          // Convert holoProbes array if provided  
          FArray holoProbes;
          if (holoProbes_array.size() > 0) {
              py::buffer_info probes_buf = holoProbes_array.request();
              float* probes_data_ptr = static_cast<float*>(probes_buf.ptr);
              holoProbes.assign(probes_data_ptr, probes_data_ptr + probes_buf.size);
          }
          
          // Convert initProbePhase array if provided
          FArray initProbePhase;
          if (initProbePhase_array.size() > 0) {
              py::buffer_info probe_phase_buf = initProbePhase_array.request();
              float* probe_phase_data_ptr = static_cast<float*>(probe_phase_buf.ptr);
              initProbePhase.assign(probe_phase_data_ptr, probe_phase_data_ptr + probe_phase_buf.size);
          }
          
          // Call the original C++ function
          F2DArray result = PhaseRetrieval::reconstruct_iter(holograms, numImages, imSize, fresnelNumbers, iterations,
                                                             initialPhase, algorithm, algoParameters, minPhase, maxPhase,
                                                             minAmplitude, maxAmplitude, support, outsideValue, padSize,
                                                             padType, padValue, projectionType, kernelType, holoProbes,
                                                             initProbePhase, calcError);
          
          // Convert F2DArray result to list of numpy arrays with proper dimensions
          py::list output_list;
          for (size_t i = 0; i < result.size(); ++i) {
              const auto& array = result[i];
              
              if (i < 3) {
                  // First 3 arrays: phase, amplitude, probe_phase (2D)
                  auto output = py::array_t<float>(rows * cols);
                  py::buffer_info output_buf = output.request();
                  float* output_ptr = static_cast<float*>(output_buf.ptr);
                  
                  // Copy data
                  std::copy(array.begin(), array.end(), output_ptr);
                  
                  // Reshape to 2D
                  output.resize({rows, cols});
                  output_list.append(output);
              } else {
                  // Last 2 arrays: step_errors, pm_errors (1D)
                  auto output = py::array_t<float>(array.size());
                  py::buffer_info output_buf = output.request();
                  float* output_ptr = static_cast<float*>(output_buf.ptr);
                  
                  // Copy data (keep as 1D)
                  std::copy(array.begin(), array.end(), output_ptr);
                  output_list.append(output);
              }
          }
          
          return output_list;
    }, "Iterative phase retrieval with auto-parsing from numpy array",
          py::arg("holograms"),
          py::arg("fresnelNumbers"),
          py::arg("iterations") = 200,
          py::arg("initialPhase") = py::array_t<float>(),
          py::arg("algorithm") = ProjectionSolver::Algorithm::AP,
          py::arg("algoParameters") = FArray(),
          py::arg("minPhase") = -1e10f,
          py::arg("maxPhase") = 1e10f,
          py::arg("minAmplitude") = 0.0f,
          py::arg("maxAmplitude") = 1e10f,
          py::arg("support") = IntArray(),
          py::arg("outsideValue") = 0.0f,
          py::arg("padSize") = IntArray(),
          py::arg("padType") = CUDAUtils::PaddingType::Replicate,
          py::arg("padValue") = 0.0f,
          py::arg("projectionType") = PMagnitudeCons::Type::Averaged,
          py::arg("kernelType") = CUDAPropKernel::Type::Fourier,
          py::arg("holoProbes") = py::array_t<float>(),
          py::arg("initProbePhase") = py::array_t<float>(),
          py::arg("calcError") = false);

    // Bind EPI reconstruction function with numpy array auto-parsing
    m.def("reconstruct_epi", [](py::array_t<float> holograms_array, const F2DArray& fresnelNumbers,
                               int iterations, py::array_t<float> initialPhase_array, py::array_t<float> initialAmplitude_array,
                               float minPhase, float maxPhase, float minAmplitude, float maxAmplitude,
                               const IntArray& support, float outsideValue, const IntArray& padSize,
                               PMagnitudeCons::Type projectionType, CUDAPropKernel::Type kernelType, bool calcError) {
          
          py::buffer_info holo_buf = holograms_array.request();
          
          // Parse dimensions from numpy array
          int numImages, rows, cols;
          if (holo_buf.ndim == 3) {
              numImages = holo_buf.shape[0];
              rows = holo_buf.shape[1];
              cols = holo_buf.shape[2];
          } else if (holo_buf.ndim == 2) {
              // Shape: (rows, cols) - single image
              numImages = 1;
              rows = holo_buf.shape[0];
              cols = holo_buf.shape[1];
          } else {
              throw std::runtime_error("Holograms array must be 2D or 3D");
          }
          
          IntArray measSize {rows, cols};
          
          if (padSize.empty()) {
              throw std::runtime_error("Padding size is required for EPI algorithm!");
          }
          IntArray imSize = {rows + 2 * padSize[0], cols + 2 * padSize[1]};
          
          // Convert holograms numpy array to FArray (flatten)
          FArray holograms;
          float* holo_data_ptr = static_cast<float*>(holo_buf.ptr);
          holograms.assign(holo_data_ptr, holo_data_ptr + holo_buf.size);
          
          // Convert initialPhase array if provided
          FArray initialPhase;
          if (initialPhase_array.size() > 0) {
              py::buffer_info phase_buf = initialPhase_array.request();
              float* phase_data_ptr = static_cast<float*>(phase_buf.ptr);
              initialPhase.assign(phase_data_ptr, phase_data_ptr + phase_buf.size);
          }
          
          // Convert initialAmplitude array if provided
          FArray initialAmplitude;
          if (initialAmplitude_array.size() > 0) {
              py::buffer_info amp_buf = initialAmplitude_array.request();
              float* amp_data_ptr = static_cast<float*>(amp_buf.ptr);
              initialAmplitude.assign(amp_data_ptr, amp_data_ptr + amp_buf.size);
          }
          
          // Call the original C++ function
          F2DArray result = PhaseRetrieval::reconstruct_epi(holograms, numImages, measSize, fresnelNumbers, iterations, imSize,
                                                            initialPhase, initialAmplitude, minPhase, maxPhase, minAmplitude, maxAmplitude,
                                                            support, outsideValue, projectionType, kernelType, calcError);
          
          // Convert F2DArray result to list of numpy arrays with proper dimensions
          py::list output_list;
          for (size_t i = 0; i < result.size(); ++i) {
              const auto& array = result[i];
              
              if (i < 2) {
                  // First 2 arrays: phase, amplitude (2D with imSize dimensions)
                  auto output = py::array_t<float>(imSize[0] * imSize[1]);
                  py::buffer_info output_buf = output.request();
                  float* output_ptr = static_cast<float*>(output_buf.ptr);
                  
                  // Copy data
                  std::copy(array.begin(), array.end(), output_ptr);
                  
                  // Reshape to 2D with measurement size
                  output.resize({imSize[0], imSize[1]});
                  output_list.append(output);
              } else {
                  // Last 2 arrays: step_errors, pm_errors (1D)
                  auto output = py::array_t<float>(array.size());
                  py::buffer_info output_buf = output.request();
                  float* output_ptr = static_cast<float*>(output_buf.ptr);
                  
                  // Copy data (keep as 1D)
                  std::copy(array.begin(), array.end(), output_ptr);
                  output_list.append(output);
              }
          }
          
          return output_list; 
    }, "EPI phase retrieval with auto-parsing from numpy array",
          py::arg("holograms"),
          py::arg("fresnelNumbers"),
          py::arg("iterations") = 200,
          py::arg("initialPhase") = py::array_t<float>(),
          py::arg("initialAmplitude") = py::array_t<float>(),
          py::arg("minPhase") = -1e10f,
          py::arg("maxPhase") = 1e10f,
          py::arg("minAmplitude") = 0.0f,
          py::arg("maxAmplitude") = 1e10f,
          py::arg("support") = IntArray(),
          py::arg("outsideValue") = 0.0f,
          py::arg("padSize") = IntArray(),
          py::arg("projectionType") = PMagnitudeCons::Type::Averaged,
          py::arg("kernelType") = CUDAPropKernel::Type::Fourier,
          py::arg("calcError") = false);

    // Bind CTFReconstructor class
    py::class_<PhaseRetrieval::CTFReconstructor>(m, "CTFReconstructor")
        .def(py::init<int, int, const IntArray&, const F2DArray&, float, float,
                      float, const IntArray&, CUDAUtils::PaddingType, float>(),
             "Initialize CTF reconstructor",
             py::arg("batchsize"), 
             py::arg("images"), 
             py::arg("imsize"), 
             py::arg("fresnelnumbers"), 
             py::arg("lowFreqLim"),
             py::arg("highFreqLim"), 
             py::arg("ratio"), 
             py::arg("padsize") = IntArray(), 
             py::arg("padtype") = CUDAUtils::PaddingType::Replicate, 
             py::arg("padvalue") = 0.0f)
        .def("reconsBatch", &PhaseRetrieval::CTFReconstructor::reconsBatch,
             "Reconstruct a batch of holograms using CTF",
             py::arg("holograms"));
    
    // Bind Reconstructor class
    py::class_<PhaseRetrieval::Reconstructor>(m, "Reconstructor")
        .def(py::init<int, int, const IntArray&, const F2DArray&, int, ProjectionSolver::Algorithm, const FArray&,
                      float, float, float, float, const IntArray&, float, const IntArray&, CUDAUtils::PaddingType,
                      float, PMagnitudeCons::Type, CUDAPropKernel::Type>(),
             "Initialize Iterative Reconstructor",
             py::arg("batchsize"),
             py::arg("images"),
             py::arg("imsize"),
             py::arg("fresnelNumbers"),
             py::arg("iter"),
             py::arg("algo"),
             py::arg("algoParams"),
             py::arg("minPhase"),
             py::arg("maxPhase"), 
             py::arg("minAmplitude"),
             py::arg("maxAmplitude"),
             py::arg("support"),
             py::arg("outsideValue"),
             py::arg("padsize") = IntArray(),
             py::arg("padtype") = CUDAUtils::PaddingType::Replicate,
             py::arg("padvalue") = 0.0f,
             py::arg("projType") = PMagnitudeCons::Type::Averaged,
             py::arg("kernelType") = CUDAPropKernel::Type::Fourier)
        .def("reconsBatch", &PhaseRetrieval::Reconstructor::reconsBatch,
             "Reconstruct a batch of holograms using iterative method",
             py::arg("holograms"),
             py::arg("initialPhase"));

}