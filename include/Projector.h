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

struct ProbeProjection
{
    WaveField projection;
    WaveField probeProjection;
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
        virtual ProbeProjection project(const WaveField& waveField, const WaveField &probeField);
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
        virtual ProbeProjection project(const WaveField& psi, const WaveField &probeField) override;
        ~PAmplitudeCons() = default;
};

class PPhaseCons: public Projector
{
    private:
        /* Phase max:inf, min:-inf */
        float maxPhase;
        float minPhase;
        float *targetPhase;
    public:
        PPhaseCons(float minPha, float maxPha): minPhase(minPha), maxPhase(maxPha) {}
        virtual Projection project(const WaveField& psi) override;
        virtual ProbeProjection project(const WaveField& psi, const WaveField &probeField) override;
        ~PPhaseCons() = default;
};

class PSupportCons: public Projector
{
    private:
        const float *support;
        cuFloatComplex *complexWave;
        float outsideValue;
    public:
        PSupportCons(const float *supp, int size, float outValue);
        virtual Projection project(const WaveField& psi) override;
        virtual ProbeProjection project(const WaveField& psi, const WaveField &probeField) override;
        ~PSupportCons();
};

class MultiObjectCons: public Projector
{
    private:
        Projector *pPhaCons;
        Projector *pAmpCons;
        Projector *pSuppCons;

    public:
        MultiObjectCons(Projector *pPha, Projector *pAmp, Projector *pSupp): pPhaCons(pPha), pAmpCons(pAmp), pSuppCons(pSupp) {}
        virtual Projection project(const WaveField& psi) override;
        virtual ProbeProjection project(const WaveField& psi, const WaveField &probeField) override;
        ~MultiObjectCons() = default;
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
        const float *p_measurements;
        IntArray imSize;
        bool calculateError;
        std::vector<PropagatorPtr> propagators;
        int numImages;
        int batchSize;
        int blockSize;
        int numBlocks;

        cuFloatComplex *complexWave;
        cuFloatComplex *cmp3DWave;
        cuFloatComplex *probeWave;
        cuFloatComplex *probe;
        float *amp3DWave;
        float residual;
        // choose function according to different projection methods
        Method calculate;
        void projectStep(const float *measuredGrams, const PropagatorPtr &prop);
        void projAveraged();
        void projSequential();
        void projCyclic();
        void projProbeAveraged();

    public:
        PMagnitudeCons(const float *measuredGrams, int numimages, const IntArray &imsize, const std::vector<PropagatorPtr> &props,
                       Type projectionType, bool calcError = true);
        PMagnitudeCons(const float *measuredGrams, const float *p_measuredGrams, int numimages, const IntArray &imsize,
                       const std::vector<PropagatorPtr> &props, Type projectionType, bool calcError = true);
        virtual Projection project(const WaveField& waveField) override;
        virtual ProbeProjection project(const WaveField& waveField, const WaveField &probeField) override;
        ~PMagnitudeCons();
};

#endif