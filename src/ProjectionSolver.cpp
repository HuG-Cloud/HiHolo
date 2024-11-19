#include <iostream>
#include "ProjectionSolver.h"

const float ProjectionSolver::terminateThreshold = -7.2;
const int ProjectionSolver::terminateIterations = 100;

ProjectionSolver::ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi, Algorithm algo, const FArray &algoParameters,
                                   bool calError): projMagnitude(PM), projObject(PS), algorithm(algo), parameters(algoParameters), psi(initialPsi),
                                   calculateError(calError), oldPsi(initialPsi), PMPsi(initialPsi)
{    
    std::unordered_map<Algorithm, Method> methodMap {{AP, &ProjectionSolver::updateStepAP}, {RAAR, &ProjectionSolver::updateStepRAAR}, 
                                                     {HIO, &ProjectionSolver::updateStepHIO}, {DRAP, &ProjectionSolver::updateStepDRAP}};
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
    residual = F2DArray(3, FArray());

}

ProjectionSolver::~ProjectionSolver() {}

IterationResult ProjectionSolver::execute(int iterations)
{   
    // print iterative algorithm
    std::cout << "Choosing algorithm: ";
    switch (algorithm) {
        case AP: std::cout << "AP"; break;
        case RAAR: std::cout << "RAAR"; break;
        case HIO: std::cout << "HIO"; break;
        case DRAP: std::cout << "DRAP"; break;
    }
    std::cout << std::endl;

    /* error measurements
       initialize the errors */
    for (int i = 0; i < residual.size(); i++) {
        if (i < 2)
            residual[i].resize(iterations, 0);
        else
            residual[i].resize(iterations, FloatInf);
    }
    
    std::cout << "Start iteration" << std::endl;
    while (!isConverged && currentIteration < iterations)
    {   
        // std::cout << "Iteration " << currentIteration << std::endl;
        /* iterate */
        update(this);

        /* Calculate Step error*/
        // if (calculateStep) {
        //     setResidual(0, MathUtils::complexL2Norm(psi.getComplexWave(), oldPsi.getComplexWave()));
        // }

        // /* Calculate Gap error */
        // if (calculateGap)
        // {
        //     auto tmpComplexWave = projObject->project(oldPsi).projection.getComplexWave();
        //     setResidual(1, MathUtils::complexL2Norm(PMPsi.getComplexWave(), tmpComplexWave));
        // }

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
    
    std::cout << "The last Iteration: " << currentIteration << std::endl;
    psi = projMagnitude->project(psi).projection;
    return {psi, residual};
}

void ProjectionSolver::updateStepAP()
{   
    Projection magnitudeResult = projMagnitude->project(psi);
    Projection objectResult = projObject->project(magnitudeResult.projection);

    setResidual(2, magnitudeResult.residual);
    PMPsi = magnitudeResult.projection;
    psi = objectResult.projection;
    
}

void ProjectionSolver::updateStepRAAR()
{
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

    setResidual(2, magnitudeResult.residual);
    PMPsi = magnitudeResult.projection;
    // update the final psi, xNew = (b/2) .* (xNew + x) + (1-b) .* xPM;
    psi = (objectResult.reflection + psi) * (b / 2.0f) + (1.0f - b) * PMPsi;

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

    setResidual(2, magnitudeResult.residual);
    PMPsi = magnitudeResult.projection;
    psi = (objectResult.reflection + psi + (1.0f - b) * PMPsi) * 0.5f;
    
}

/* Dougles-Rachford Alternating Projections */
void ProjectionSolver::updateStepDRAP()
{
    if (parameters.size() != 1) {
        throw std::invalid_argument("Incorrect parameter setting!");
    }
    float b = parameters[0];

    // x_n+1 = PS((1-b) * PM(x) - b * x) - b * (PM(x) - x)
    Projection magnitudeResult = projMagnitude->project(psi);
    Projection objectResult = projObject->project((1.0f + b) * magnitudeResult.projection - b * psi);

    setResidual(2, magnitudeResult.residual);
    PMPsi = magnitudeResult.projection;
    psi = objectResult.projection - (PMPsi - psi) * b;

}

// Each index represents the different errors
void ProjectionSolver::setResidual(int index, float error)
{
    residual[index][currentIteration] = error;
}
