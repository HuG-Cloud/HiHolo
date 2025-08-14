#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <vector>
#include <complex>

typedef std::vector<int> IntArray;
typedef std::vector<IntArray> Int2DArray;

typedef std::vector<float> FArray;
typedef std::vector<FArray> F2DArray;

typedef std::vector<double> DArray;
typedef std::vector<DArray> D2DArray;

typedef std::vector<std::complex<float>> ComArray;
typedef std::vector<ComArray> Com2DArray;

typedef std::vector<bool> BArray;
typedef std::vector<BArray> B2DArray;

typedef std::vector<uint16_t> U16Array;
typedef std::vector<uint8_t> U8Array;

#endif