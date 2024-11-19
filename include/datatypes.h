#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <vector>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>

typedef std::vector<int> IntArray;

typedef std::vector<float> FArray;
typedef std::vector<FArray> F2DArray;

typedef std::vector<std::complex<float>> ComArray;
typedef std::vector<ComArray> Com2DArray; 

typedef std::vector<bool> BArray;
typedef std::vector<BArray> B2DArray;

#endif