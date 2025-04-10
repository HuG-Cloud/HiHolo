#include <iostream>
#include "ProjectionSolver.h"

const float ProjectionSolver::terminateThreshold = -7.2;
const int ProjectionSolver::terminateIterations = 100;

ProjectionSolver::ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi, Algorithm algo, const FArray &algoParameters,
                                   bool calError): projMagnitude(PM), projObject(PS), algorithm(algo), parameters(algoParameters),
                                   psi(initialPsi), calculateError(calError), oldPsi(initialPsi)
{    
    // Map holographic algorithm to corresponding update method
    std::unordered_map<Algorithm, Method> methodMap {{AP, &ProjectionSolver::updateStepAP}, {RAAR, &ProjectionSolver::updateStepRAAR}, 
                                                     {HIO, &ProjectionSolver::updateStepHIO}, {DRAP, &ProjectionSolver::updateStepDRAP},
                                                     {BIPEPI, &ProjectionSolver::updateStepAP}};
                                                     
    auto iterator = methodMap.find(algorithm);
    if (iterator != methodMap.end()) {
        update = iterator->second;
    } else {
        throw std::invalid_argument("Invalid algorithm!");
    }
    
    currentIteration = 1;
    isConverged = false;
    /* error measurements
       initialize the errors */
    if (calculateError)
        residual = F2DArray(2, FArray());
}

ProjectionSolver::ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi, const WaveField &initialProbe,
                                   bool calError): projMagnitude(PM), projObject(PS), algorithm(APWP), psi(initialPsi),
                                   probe(initialProbe), calculateError(calError), oldPsi(initialPsi)
{
    update = &ProjectionSolver::updateStepAPWP;
    currentIteration = 1;
    isConverged = false;
    /* error measurements
       initialize the errors */
    if (calculateError)
        residual = F2DArray(2, FArray());
}

IterationResult ProjectionSolver::execute(int iterations)
{   
    /* error measurements
       initialize the errors */
    if (calculateError) {
        residual[0].resize(iterations, 0);
        residual[1].resize(iterations, FloatInf);
    }
    
    // Iterate until convergence or maximum iterations
    while (!isConverged && currentIteration < iterations)
    {   
        // std::cout << "Iteration " << currentIteration << std::endl;
        /* iterate */
        update(this);

        /* Calculate Step error*/
        if (calculateError) {
            setResidual(0, CUDAUtils::computeL2Norm(psi.getComplexWave(), oldPsi.getComplexWave(), psi.getSize()));
        }

        // /* test if iteration is converged */
        // if (calculateStep && (currentIteration > 10 * terminateIterations))
        // {
        //     std::cout << "Testing if iteration is converged: " << std::endl;
        //     DArray lastSlice(residual[0].begin() + currentIteration - 10 * terminateIterations + 1, 
        //                      residual[0].begin() + currentIteration + 1);

        //     lastSlice = MathUtils::differVector(MathUtils::movmean(MathUtils::medfilt1(lastSlice), terminateIterations));
        //     for (auto &element: lastSlice) {
        //         element = std::abs(element);
        //     }

        //     bool abortStep = std::log10(MathUtils::movmean(lastSlice, 50).back()) < terminateThreshold;
        //     bool abortPM = residual[2][currentIteration] < std::pow(10.0, terminateThreshold);
        //     if (abortStep || abortPM)
        //     {
        //         std::cout << "Reach the condition for convergence and skip the remaining iterations!" << std::endl;
        //         isConverged = true;
        //     }
            
        // }
        
        oldPsi = psi;
        currentIteration++;
    }
    
    auto magnitudeResult = projMagnitude->project(psi);
    auto objectResult = projObject->project(magnitudeResult.projection);
    if (algorithm == BIPEPI) {
        psi = objectResult.projection;
    } else {
        psi = magnitudeResult.projection;
    }
    
    if (calculateError) {
        setResidual(1, magnitudeResult.residual);
        setResidual(0, CUDAUtils::computeL2Norm(psi.getComplexWave(), oldPsi.getComplexWave(), psi.getSize()));
    }
    
    return {psi, probe, residual};
}

/* Alternating Projection Algorithm */
void ProjectionSolver::updateStepAP()
{   
    Projection magnitudeResult = projMagnitude->project(psi);
    Projection objectResult = projObject->project(magnitudeResult.projection);

    if (calculateError)
        setResidual(1, magnitudeResult.residual);
    psi = objectResult.projection;
}

/* Alternating Projection Algorithm with Probe */
void ProjectionSolver::updateStepAPWP()
{
    ProbeProjection magnitudeResult = projMagnitude->project(psi, probe);
    Projection objectResult = projObject->project(magnitudeResult.projection);

    if (calculateError)
        setResidual(1, magnitudeResult.residual);
    probe = magnitudeResult.probeProjection;
    psi = objectResult.projection;
}

/* Relaxed Averaged Alternating Reflections Algorithm */
void ProjectionSolver::updateStepRAAR()
{
    // Check parameters size and read parameters
    if (parameters.size() != 3) {
        throw std::invalid_argument("Incorrect parameter setting!");
    }

    float b0 = parameters[0];
    float bM = parameters[1];
    float bS = parameters[2];
    // calculate relaxation parameter
    float expTerm = std::exp(-std::pow(currentIteration / bS, 3.0f));
    float b = expTerm * b0 + (1.0f - expTerm) * bM;

    Reflection magnitudeResult = projMagnitude->reflect(psi);
    Reflection objectResult = projObject->reflect(magnitudeResult.reflection);

    if (calculateError)
        setResidual(1, magnitudeResult.residual);
    // update the final psi, xNew = (b/2) .* (xNew + x) + (1-b) .* xPM;
    // psi = (objectResult.reflection + psi) * (b / 2.0f) + (1.0f - b) * PMPsi;
    (psi + objectResult.reflection) * (b / 2.0f) + magnitudeResult.projection * (1.0f - b);
}

/* Hybrid Input-Output Algorithm */
void ProjectionSolver::updateStepHIO()
{
    if (parameters.size() != 1) {
        throw std::invalid_argument("Incorrect parameter setting!");
    }
    float b = parameters[0];

    // x_n+1 = (RS(RM(x) + (b-1)*PM(x)) + x + (1-b)*PM(x)) * 0.5 
    Projection magnitudeResult = projMagnitude->project(psi);
    Reflection objectResult = projObject->reflect((1.0f + b) * magnitudeResult.projection - psi);

    if (calculateError)
        setResidual(1, magnitudeResult.residual);
    // psi = (objectResult.reflection + psi + (1.0f - b) * PMPsi) * 0.5f;
    ((psi + objectResult.reflection) + magnitudeResult.projection * (1.0f - b)) * 0.5f;
}

/* Dougles-Rachford Alternating Projections Algorithm */
void ProjectionSolver::updateStepDRAP()
{
    if (parameters.size() != 1) {
        throw std::invalid_argument("Incorrect parameter setting!");
    }
    float b = parameters[0];

    // x_n+1 = PS((1-b) * PM(x) - b * x) - b * (PM(x) - x)
    Projection magnitudeResult = projMagnitude->project(psi);
    Projection objectResult = projObject->project((1.0f + b) * magnitudeResult.projection - b * psi);

    if (calculateError)
        setResidual(1, magnitudeResult.residual);
    psi = objectResult.projection - (magnitudeResult.projection - psi) * b;
}

// Each index represents the different errors
void ProjectionSolver::setResidual(int index, float error)
{
    residual[index][currentIteration - 1] = error;
}