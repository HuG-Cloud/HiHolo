#include "Projector.h"

// An empty implementation because of the actual call of derived class
Projection Projector::project(const WaveField& waveField)
{
    return {waveField, FloatInf};
}

ProbeProjection Projector::project(const WaveField& waveField, const WaveField &probeField)
{
    return {waveField, probeField, FloatInf};
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
    cudaFree(targetAmplitude);
    return {updatedPsi, residual};
}

ProbeProjection PAmplitudeCons::project(const WaveField& psi, const WaveField &probeField)
{
    return {psi, probeField, FloatInf};
}

Projection PPhaseCons::project(const WaveField& psi)
{
    if (maxPhase < minPhase) {
        throw std::invalid_argument("maxPhase can not be less than minPhase");
    }

    float residual = FloatInf;

    // without any phase constraints
    if (maxPhase == FloatInf && minPhase == -FloatInf)
        return {psi, residual};
    
    int rows = psi.getRows();
    int cols = psi.getColumns();
    cudaMalloc(&targetPhase, rows * cols * sizeof(float));

    int blockSize = 1024;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;

    // update phase according to max/min constraints
    psi.getPhase(targetPhase);
    adjustPhase<<<gridSize, blockSize>>>(targetPhase, maxPhase, minPhase, rows * cols);

    WaveField updatedPsi(psi);
    updatedPsi.setByPhase(targetPhase);
    cudaFree(targetPhase);
    return {updatedPsi, residual};
}

ProbeProjection PPhaseCons::project(const WaveField& psi, const WaveField &probeField)
{
    return {psi, probeField, FloatInf};
}

PSupportCons::PSupportCons(const float *supp, int size, float outValue):
                           support(supp), outsideValue(outValue), complexWave(nullptr)
{
    if (support) {
        cudaMalloc(&complexWave, sizeof(cuFloatComplex) * size);
    }
}

Projection PSupportCons::project(const WaveField& psi)
{
    float residual = FloatInf;

    if (!support)
        return {psi, residual};
    
    int rows = psi.getRows();
    int cols = psi.getColumns();
    int blockSize = 1024;
    int numBlocks = (rows * cols + blockSize - 1) / blockSize;

    psi.getComplexWave(complexWave);
    adjustComplexWave<<<numBlocks, blockSize>>>(complexWave, support, outsideValue, rows * cols);
    
    WaveField updatedPsi(rows, cols, complexWave);
    return {updatedPsi, residual};
}

ProbeProjection PSupportCons::project(const WaveField& psi, const WaveField &probeField)
{
    return {psi, probeField, FloatInf};
}

PSupportCons::~PSupportCons()
{
    if (complexWave)
        cudaFree(complexWave);
}    

Projection MultiObjectCons::project(const WaveField& psi)
{
    Projection phaConsResult = pPhaCons->project(psi);
    Projection ampConsResult = pAmpCons->project(phaConsResult.projection);
    Projection suppConsResult = pSuppCons->project(ampConsResult.projection);

    return suppConsResult;
}

ProbeProjection MultiObjectCons::project(const WaveField& psi, const WaveField &probeField)
{
    return {psi, probeField, FloatInf};
}

int PMagnitudeCons::currentIteration = 0;

PMagnitudeCons::PMagnitudeCons(const float *measuredGrams, int numimages, const IntArray &imsize, const std::vector<PropagatorPtr> &props,
                       Type projectionType, bool calcError): measurements(measuredGrams), numImages(numimages), imSize(imsize), propagators(props), 
                       type(projectionType), calculateError(calcError), p_measurements(nullptr)
{   
    // Check projection type and set batch size
    if (type == Averaged) {
        batchSize = numImages;
    } else if (type == Sequential || type == Cyclic) {
        batchSize = 1;
    } else {
        throw std::invalid_argument("Invalid projection computing method!");
    }

    // Map projection type to corresponding method
    std::unordered_map<Type, Method> methodMap {{Averaged, &PMagnitudeCons::projAveraged}, {Sequential, &PMagnitudeCons::projSequential},
                                                {Cyclic, &PMagnitudeCons::projCyclic}};
    auto iterator = methodMap.find(type);
    calculate = iterator->second;

    cudaMalloc(&complexWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1]);
    cudaMalloc(&cmp3DWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1] * batchSize);
    cudaMalloc(&amp3DWave, sizeof(float) * imSize[0] * imSize[1] * batchSize);

    blockSize = 1024;
    numBlocks = (imSize[0] * imSize[1] * batchSize + blockSize - 1) / blockSize;
}

PMagnitudeCons::PMagnitudeCons(const float *measuredGrams, const float *p_measuredGrams, int numimages, const IntArray &imsize, 
                               const std::vector<PropagatorPtr> &props, Type projectionType, bool calcError): measurements(measuredGrams), imSize(imsize), 
                               p_measurements(p_measuredGrams), numImages(numimages), propagators(props), type(projectionType), calculateError(calcError)
{
    if (type != Averaged) {
        throw std::invalid_argument("Invalid projection computing method!");
    }

    batchSize = numImages;
    calculate = &PMagnitudeCons::projProbeAveraged;

    cudaMalloc(&complexWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1]);
    cudaMalloc(&cmp3DWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1] * batchSize);
    cudaMalloc(&probeWave, sizeof(cuFloatComplex) * imSize[0] * imSize[1]);
    cudaMalloc(&probe, sizeof(cuFloatComplex) * imSize[0] * imSize[1]);
    cudaMalloc(&amp3DWave, sizeof(float) * imSize[0] * imSize[1] * batchSize);

    blockSize = 1024;
}

// Project step: propagate wavefield and constrain amplitude
void PMagnitudeCons::projectStep(const float *measuredGrams, const PropagatorPtr &prop)
{
    prop->propagate(complexWave, cmp3DWave);
    computeAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, imSize[0] * imSize[1] * batchSize);

    /* optionally calculate residual */
    if (calculateError) {
        residual = CUDAUtils::computeL2Norm(amp3DWave, measuredGrams, imSize[0] * imSize[1] * batchSize);
    } else {
        residual = FloatInf;
    }

    limitAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, measuredGrams, imSize[0] * imSize[1] * batchSize);
    prop->backPropagate(cmp3DWave, complexWave);
}

void PMagnitudeCons::projProbeAveraged()
{
    int numBlocks1 = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    int numBlocks2 = (imSize[0] * imSize[1] * batchSize + blockSize - 1) / blockSize;

    // Update probe wavefield and propagate
    numBlocks = numBlocks1;
    multiplyWaveField<<<numBlocks, blockSize>>>(probeWave, complexWave, probe, imSize[0] * imSize[1]);
    propagators[0]->propagate(probeWave, cmp3DWave);

    numBlocks = numBlocks2;
    computeAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, imSize[0] * imSize[1] * batchSize);
    if (calculateError) {
        residual = CUDAUtils::computeL2Norm(amp3DWave, measurements, imSize[0] * imSize[1] * batchSize);
    } else {
        residual = FloatInf;
    }
    limitAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, amp3DWave, measurements, imSize[0] * imSize[1] * batchSize);
    propagators[0]->backPropagate(cmp3DWave, probeWave);

    // Isolate probe and propagate
    numBlocks = numBlocks1;
    scaleComplexData<<<numBlocks, blockSize>>>(probeWave, imSize[0] * imSize[1], 1.0f / batchSize);
    updateDM<<<numBlocks, blockSize>>>(probe, probeWave, complexWave, imSize[0] * imSize[1]);
    propagators[0]->propagate(probe, cmp3DWave);

    numBlocks = numBlocks2;
    limitAmplitude<<<numBlocks, blockSize>>>(cmp3DWave, p_measurements, imSize[0] * imSize[1] * batchSize);
    propagators[0]->backPropagate(cmp3DWave, probe);

    // Isolate object wavefield from probe wavefield
    numBlocks = numBlocks1;
    scaleComplexData<<<numBlocks, blockSize>>>(probe, imSize[0] * imSize[1], 1.0f / batchSize);
    updateDM<<<numBlocks, blockSize>>>(complexWave, probeWave, probe, imSize[0] * imSize[1]);
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

ProbeProjection PMagnitudeCons::project(const WaveField &waveField, const WaveField &probeField)
{
    waveField.getComplexWave(complexWave);
    probeField.getComplexWave(probe);
    calculate(this);

    WaveField newField(imSize[0], imSize[1], complexWave);
    WaveField newProbe(imSize[0], imSize[1], probe);
    return {newField, newProbe, residual};
}

PMagnitudeCons::~PMagnitudeCons()
{
    cudaFree(complexWave);
    cudaFree(cmp3DWave);
    cudaFree(amp3DWave);

    if (p_measurements) {
        cudaFree(probeWave);
        cudaFree(probe);
    }
}