#include "Projector.h"

// An empty implementation because of the actual call of derived class
Projection Projector::project(const WaveField& waveField)
{
    return {waveField, FloatInf};
}

// Reflection on object and measurment plane is identical
Reflection Projector::reflect(const WaveField& psi)
{
    Projection proj = project(psi);
    int rows = psi.getRows();
    int cols = psi.getColumns();

    WaveField reflectedPsi(proj.projection);

    int blockSize = 1024;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;
    reflectWaveField<<<gridSize, blockSize>>>(reflectedPsi.getComplexWave(), psi.getComplexWave(), rows * cols);
    
    return {proj.projection, reflectedPsi, proj.residual};
}

Projection PAmplitudeCons::project(const WaveField& psi)
{
    if (maxAmplitude < minAmplitude) {
        throw std::invalid_argument("maxAmplitude can not be less than minAmplitude");
    }

    // residual
    float residual = FloatInf;

    // without any amplitude constraints
    if (maxAmplitude == FloatInf && minAmplitude <= 0)
        return {psi, residual};
    
    int rows = psi.getRows();
    int cols = psi.getColumns();
    cudaMalloc(&targetAmplitude, rows * cols * sizeof(float));

    int blockSize = 1024;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;

    // update amplitude according to max/min constraints
    if (maxAmplitude == minAmplitude) {
        initializeData<<<gridSize, blockSize>>>(targetAmplitude, maxAmplitude, rows * cols);
    } else {
        psi.getAmplitude(targetAmplitude);
        adjustAmplitude<<<gridSize, blockSize>>>(targetAmplitude, maxAmplitude, minAmplitude, rows * cols);
    }

    WaveField updatedPsi(psi);
    updatedPsi.setByAmplitude(targetAmplitude);
    return {updatedPsi, residual};
}

PAmplitudeCons::~PAmplitudeCons()
{
    if (targetAmplitude)
        cudaFree(targetAmplitude);
}

int PMagnitudeCons::currentIteration = 0;

PMagnitudeCons::PMagnitudeCons(const float *measuredGrams, int numimages, const IntArray &imsize, const F2DArray &fresnelnumbers,
                       Type projectionType, CUDAPropKernel::Type kernelType, bool calcError): measurements(measuredGrams), numImages(numimages), 
                       imSize(imsize), fresnelNumbers(fresnelnumbers), type(projectionType), calculateError(calcError)
{   
    // Initialize fresnel propagator(s)
    std::cout << "Choosing kernel type: ";
    switch (kernelType) {
        case CUDAPropKernel::Fourier:
            std::cout << "Fourier";
            break;
        case CUDAPropKernel::Chirp:
            std::cout << "Chirp";
            break;
        case CUDAPropKernel::ChirpLimited:
            std::cout << "ChirpLimited";
            break;
        default: 
            throw std::invalid_argument("Invalid kernel type!");
    }
    std::cout << std::endl;

    if (type == Averaged) {
        batchSize = numImages;
        propagators.push_back(std::make_unique<Propagator>(imSize, fresnelNumbers, kernelType));
    } else if (type == Sequential || type == Cyclic) {
        batchSize = 1;
        for (const auto &fNumber: fresnelNumbers) {
            F2DArray singleFresnel {fNumber};
            propagators.push_back(std::make_unique<Propagator>(imSize, singleFresnel, kernelType));
        }
    } else {
        throw std::invalid_argument("Invalid projection computing method!");
    }

    std::unordered_map<Type, Method> methodMap {{Averaged, &PMagnitudeCons::projAveraged}, {Sequential, &PMagnitudeCons::projSequential},
                                                {Cyclic, &PMagnitudeCons::projCyclic}};
    auto iterator = methodMap.find(type);
    calculate = iterator->second;
    
    std::cout << "Choosing projection method: ";
    switch (type) {
        case Averaged: std::cout << "Averaged"; break;
        case Sequential: std::cout << "Sequential"; break;
        case Cyclic: std::cout << "Cyclic"; break;
    }
    std::cout << std::endl;

    cudaMalloc(&complexWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1]);
    cudaMalloc(&cmp3DWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1] * batchSize);
    cudaMalloc(&amp3DWave, sizeof(float) * imSize[0] * imSize[1] * batchSize);

    blockSize = 1024;
    numBlocks = (imSize[0] * imSize[1] * batchSize + blockSize - 1) / blockSize;
}

void PMagnitudeCons::projectStep(const float *measuredGrams, const PropagatorPtr &prop)
{
    prop->propagate(complexWave, cmp3DWave);
    computeAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, imSize[0] * imSize[1] * batchSize);

    /* optionally calculate residual */
    // if (calculateError) {

    //     double squaredSum = 0;
    //     for (int i = 0; i < amp3DWave.size(); i++) {
    //         squaredSum += MathUtils::diffInnerProduct(amp3DWave[i], measuredGrams[i]);
    //     }
    //     residual = std::sqrt(squaredSum);

    // } else {
    //     residual = DoubleInf;
    // }

    limitAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, measuredGrams, imSize[0] * imSize[1] * batchSize);
    prop->backPropagate(cmp3DWave, complexWave);
}

void PMagnitudeCons::projAveraged()
{   
    projectStep(measurements, propagators[0]);
    scaleComplexData<<<numBlocks, blockSize>>>(complexWave, imSize[0] * imSize[1], 1.0f / batchSize);
}

void PMagnitudeCons::projSequential()
{
    for (int i = 0; i < numImages; i++) {
        projectStep(measurements + i * imSize[0] * imSize[1], propagators[i]);
    }
}

void PMagnitudeCons::projCyclic()
{
    int index = currentIteration % numImages;
    projectStep(measurements + index * imSize[0] * imSize[1], propagators[index]);
}

Projection PMagnitudeCons::project(const WaveField &waveField)
{   
    waveField.getComplexWave(complexWave);
    calculate(this);
    currentIteration++;
    
    WaveField newField(imSize[0], imSize[1], complexWave);
    return {newField, residual};
}

PMagnitudeCons::~PMagnitudeCons() {
    cudaFree(complexWave);
    cudaFree(cmp3DWave);
    cudaFree(amp3DWave);
}