#ifndef PROJECTIONSOLVER_H_
#define PROJECTIONSOLVER_H_

#include "Projector.h"

struct IterationResult
{
    const WaveField &reconsPsi;
    const WaveField &reconsProbe;
    const F2DArray &finalError;
};

class ProjectionSolver
{   
    public:
        enum Algorithm {AP, RAAR, HIO, DRAP, APWP, BIPEPI};
        typedef std::function<void(ProjectionSolver*)> Method;

    private:
        Projector *projMagnitude;
        Projector *projObject;
        bool isConverged;
        // Each FArray represents a different error
        F2DArray residual;
        WaveField psi;
        WaveField oldPsi;
        WaveField probe;

        /**
         * Whether to calculate errors including step, magnitude error
         * Step: |x_n - x_(n-1)|
         * Magnitude
         */
        bool calculateError; 
        static const float terminateThreshold;
        static const int terminateIterations;
        void setResidual(int index, float error);

        // Choose function according to algorithm
        Algorithm algorithm;
        Method update;
        void updateStepAP();
        void updateStepHIO();
        void updateStepRAAR();
        void updateStepDRAP();
        void updateStepAPWP();
        // Start at 1
        int currentIteration;
        // Parameters for RAAR/HIO/DRAP algorithm        
        FArray parameters;

    public:
        ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi,
                         Algorithm algo, const FArray &algoParameters, bool calError = true);
        ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi,
                         const WaveField &initialProbe, bool calError = true);
        IterationResult execute(int iterations);
        ~ProjectionSolver() = default;
};

#endif
