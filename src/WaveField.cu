#include "WaveField.h"

WaveField::WaveField(int in_rows, int in_cols, const cuFloatComplex *cmpWave): rows(in_rows), cols(in_cols)
{
    cudaMalloc(&complexWave, rows * cols * sizeof(cuFloatComplex));
    cudaMemcpy(complexWave, cmpWave, rows * cols * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    blockSize = 1024;
    gridSize = (rows * cols + blockSize - 1) / blockSize;
}

WaveField::WaveField(const WaveField &waveField): rows(waveField.rows), cols(waveField.cols)
{
    cudaMalloc(&complexWave, rows * cols * sizeof(cuFloatComplex));
    cudaMemcpy(complexWave, waveField.complexWave, rows * cols * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    blockSize = waveField.blockSize;
    gridSize = waveField.gridSize;
}

void WaveField::getAmplitude(float *amplitude) const
{
    computeAmplitude<<<gridSize, blockSize>>>(complexWave, amplitude, rows * cols);
}

void WaveField::getPhase(float *phase) const
{
    computePhase<<<gridSize, blockSize>>>(complexWave, phase, rows * cols);
}

void WaveField::getComplexWave(cuFloatComplex *cmpWave) const
{
    cudaMemcpy(cmpWave, complexWave, rows * cols * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
}

void WaveField::setByAmplitude(const float *targetAmplitude)
{
    setAmplitude<<<gridSize, blockSize>>>(complexWave, targetAmplitude, rows * cols);
}

void WaveField::setByPhase(const float *targetPhase)
{
    setPhase<<<gridSize, blockSize>>>(complexWave, targetPhase, rows * cols);
}

WaveField WaveField::operator+(const WaveField &waveField) const
{
    if ((rows != waveField.getRows()) || (cols != waveField.getColumns())) {
        throw std::invalid_argument("The sizes of the 2 wave fields do not match!");
    }

    WaveField newField(waveField);
    addWaveField<<<gridSize, blockSize>>>(newField.complexWave, complexWave, rows * cols);

    return newField;
}

WaveField WaveField::operator-(const WaveField &waveField) const
{
    if ((rows != waveField.getRows()) || (cols != waveField.getColumns())) {
        throw std::invalid_argument("The sizes of the 2 wave fields do not match!");
    }

    WaveField newField(*this);
    subWaveField<<<gridSize, blockSize>>>(newField.complexWave, waveField.complexWave, rows * cols);
    
    return newField;
}

WaveField &WaveField::operator*(float n)
{
    scaleComplexData<<<gridSize, blockSize>>>(complexWave, rows * cols, n);
    return *this;
}

WaveField operator*(float n, const WaveField &waveField)
{   
    WaveField newField(waveField);
    scaleComplexData<<<newField.gridSize, newField.blockSize>>>(newField.complexWave, newField.rows * newField.cols, n);
    return newField;
}

WaveField &WaveField::operator=(const WaveField &waveField)
{
    if (this == &waveField) {
        return *this;
    }

    cudaMemcpy(complexWave, waveField.complexWave, rows * cols * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    return *this;
}


std::ostream &operator<<(std::ostream &os, const WaveField &waveField)
{
    int rows = waveField.getRows();
    int cols = waveField.getColumns();

    cuFloatComplex *h_complexWave = new cuFloatComplex[rows * cols];
    cudaMemcpy(h_complexWave, waveField.complexWave, rows * cols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            os << "(" << h_complexWave[i * cols + j].x << ", " << h_complexWave[i * cols + j].y << ") ";
        }
        os << std::endl;
    }
    
    delete[] h_complexWave;
    return os;
    
}