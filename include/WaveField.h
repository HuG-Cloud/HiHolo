#ifndef WAVEFIELD_H_
#define WAVEFIELD_H_

#include <iostream>

#include "datatypes.h"
#include "cuda_utils.h"

class WaveField
{
    private:
        int rows, cols;
        int blockSize;
        int gridSize;
        // Polar representation, matrix is represented by 1D
        cuFloatComplex *complexWave;

    public:
        WaveField(int in_rows, int in_cols, const cuFloatComplex *cmpWave);
        WaveField(const WaveField &waveField);
        ~WaveField() {cudaFree(complexWave);}
        void getAmplitude(float *amplitude) const;
        void getPhase(float *phase) const;
        int getRows() const {return rows;}
        int getColumns() const {return cols;} 
        void getComplexWave(cuFloatComplex *cmpWave) const;
        cuFloatComplex *getComplexWave() const {return complexWave;}
        void setByAmplitude(const float *targetAmplitude);
        void setByPhase(const float *targetPhase);
        WaveField operator+(const WaveField &waveField) const;
        WaveField operator-(const WaveField &waveField) const;
        WaveField &operator*(float n);
        WaveField &operator=(const WaveField &waveField);
        friend WaveField operator*(float n, const WaveField &waveField);
        friend std::ostream &operator<<(std::ostream &os, const WaveField &waveField);
};

#endif