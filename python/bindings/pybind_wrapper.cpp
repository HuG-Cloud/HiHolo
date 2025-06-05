#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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

PYBIND11_MODULE(fastholo, m) {
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
        .value("BIPEPI", ProjectionSolver::Algorithm::BIPEPI);

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
    
    // 绑定距离标定函数
    m.def("calibrateDistance", &ImageUtils::calibrateDistance,
          "Calibrate distance by standard method",
          py::arg("holograms"),
          py::arg("numImages"),
          py::arg("rows"),
          py::arg("cols"),
          py::arg("length"),
          py::arg("pixelSize"),
          py::arg("nz"),
          py::arg("stepSize"));
    
    // 绑定CTF重建函数
    m.def("reconstruct_ctf", &PhaseRetrieval::reconstruct_ctf,
          "CTF (Contrast Transfer Function) based phase retrieval",
          py::arg("holograms"), 
          py::arg("numImages"), 
          py::arg("imSize"), 
          py::arg("fresnelnumbers"), 
          py::arg("lowFreqLim"), 
          py::arg("highFreqLim"),
          py::arg("betaDeltaRatio"), 
          py::arg("padSize") = IntArray(),
          py::arg("padType") = CUDAUtils::PaddingType::Replicate, 
          py::arg("padValue") = 0.0f);
    
    // 绑定iterative重建函数
    m.def("reconstruct_iter", &PhaseRetrieval::reconstruct_iter,
          "Iterative phase retrieval",
          py::arg("holograms"),
          py::arg("numImages"),
          py::arg("imSize"),
          py::arg("fresnelNumbers"),
          py::arg("iterations"),
          py::arg("initialPhase"),
          py::arg("algorithm"),
          py::arg("algoParameters"),
          py::arg("minPhase"),
          py::arg("maxPhase"),
          py::arg("minAmplitude"),
          py::arg("maxAmplitude"),
          py::arg("support"),
          py::arg("outsideValue"),
          py::arg("padSize"),
          py::arg("padType"),
          py::arg("padValue"),
          py::arg("projectionType"),
          py::arg("kernelType"),
          py::arg("holoProbes"),
          py::arg("initProbePhase"),
          py::arg("calcError"));

    m.def("reconstruct_bipepi", &PhaseRetrieval::reconstruct_bipepi,
          "Bipepi phase retrieval",
          py::arg("holograms"),
          py::arg("numImages"),
          py::arg("measSize"),
          py::arg("fresnelNumbers"),
          py::arg("iterations"),
          py::arg("imSize"),
          py::arg("initialPhase"),
          py::arg("initialAmplitude"),
          py::arg("minPhase"),
          py::arg("maxPhase"),
          py::arg("minAmplitude"),
          py::arg("maxAmplitude"),
          py::arg("support"),
          py::arg("outsideValue"),
          py::arg("projectionType"),
          py::arg("kernelType"),
          py::arg("calcError"));

    // 绑定CTFReconstructor类
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
    
    // 绑定Reconstructor类
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