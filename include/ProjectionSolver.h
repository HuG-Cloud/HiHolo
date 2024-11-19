#ifndef PROJECTIONSOLVER_H_
#define PROJECTIONSOLVER_H_

#include <functional>
#include <unordered_map>

#include "Projector.h"
#include "math_utils.h"

struct IterationResult
{
    const WaveField &reconsPsi;
    const F2DArray &finalError;
};

class ProjectionSolver
{   
    public:
        enum Algorithm{AP, RAAR, HIO, DRAP};     
        typedef std::function<void(ProjectionSolver*)> Method;

    private:
        Projector *projMagnitude;
        Projector *projObject;
        bool isConverged;
        // each Frray represents a different error
        F2DArray residual;
        WaveField psi;
        WaveField oldPsi;
        WaveField PMPsi;

        /**
         * whether to calculate errors including step, gap, magnitude error
         * Step: |x_n - x_(n-1)|
         * Gap: |PS(x) - PM(x)| / N
         * Magnitude
         */
        bool calculateError; 
        static const float terminateThreshold;
        static const int terminateIterations;
        void setResidual(int index, float error);

        // choose function according to algorithm
        Algorithm algorithm;
        Method update;
        void updateStepAP();
        void updateStepHIO();
        void updateStepRAAR();
        void updateStepDRAP();

        // start at 1
        int currentIteration;
        // parameters for RAAR/HIO/DRAP algorithm        
        FArray parameters;

    public:
        ProjectionSolver(Projector *PM, Projector *PS, const WaveField &initialPsi, Algorithm algo, const FArray &algoParameters, bool calError);
        IterationResult execute(int iterations);
        ~ProjectionSolver();
};

#endif
