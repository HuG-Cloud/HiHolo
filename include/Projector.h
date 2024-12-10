#ifndef PROJECTOR_H_
#define PROJECTOR_H_

#include <memory>
#include <functional>
#include <unordered_map>

#include "WaveField.h"
#include "math_utils.h"
#include "Propagator.h"

struct Projection
{
    WaveField projection;
    float residual;
};

struct Reflection
{
    WaveField projection;
    WaveField reflection;
    float residual;
};

class Projector
{
    public:
        Projector() = default;
        virtual Projection project(const WaveField& waveField);
        Reflection reflect(const WaveField& waveField);
        virtual ~Projector() = default;
};

class PAmplitudeCons: public Projector
{
    private:
        /* Amplitude max:inf, min:0 */
        float maxAmplitude;
        float minAmplitude;
        float *targetAmplitude;
    public:
        PAmplitudeCons(float minAmp, float maxAmp): minAmplitude(minAmp), maxAmplitude(maxAmp) {}
        virtual Projection project(const WaveField& psi) override;
        ~PAmplitudeCons();
};

class PMagnitudeCons: public Projector
{
    public:
        // Represents different methods of calculating projections
        enum Type {Averaged, Sequential, Cyclic};
        typedef std::function<void(PMagnitudeCons*)> Method;
        
    private:
        static int currentIteration;
        Type type;
        F2DArray fresnelNumbers;
        const float *measurements;
        IntArray imSize;
        bool calculateError;
        std::vector<PropagatorPtr> propagators;
        int numImages;
        int batchSize;
        int blockSize;
        int numBlocks;

        cuFloatComplex *complexWave;
        cuFloatComplex *cmp3DWave;
        float *amp3DWave;
        float residual;
        // choose function according to different projection methods
        Method calculate;
        void projectStep(const float *measuredGrams, const PropagatorPtr &prop);
        void projAveraged();
        void projSequential();
        void projCyclic();

    public:
        PMagnitudeCons(const float *measuredGrams, int numimages, const IntArray &imsize, const std::vector<PropagatorPtr> &props,
                       Type projectionType, bool calcError = false);
        virtual Projection project(const WaveField& waveField) override;
        ~PMagnitudeCons();
};

#endif