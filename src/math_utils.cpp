#include "math_utils.h"
#include <iostream>

const std::complex<float> i_c = std::complex<float>(0, 1);
const uint16_t maxUInt_16 = std::numeric_limits<uint16_t>::max();
const float FloatInf = std::numeric_limits<float>::infinity();

FArray MathUtils::genEquidisRange(float start, float end, int n)
{
    if (n <= 1 || start > end)
        throw std::invalid_argument("Invalid arguements for generating equidistant range!");
    
    FArray range;
    float delta = (end - start) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        range.push_back(start + i * delta);
    }
    
    return range;
}

float MathUtils::complexL2Norm(const ComArray &vec1, const ComArray &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++)
    {   
        sum += std::norm(vec1[i] - vec2[i]);
    }

    return std::sqrt(sum);
}

std::complex<float> MathUtils::getInitCoeff(const FArray &vec)
{
    ComArray tempFreq(vec.size());
    for (int i = 0; i < tempFreq.size(); i++) {
        float tmp = std::sqrt(std::abs(vec[i]));
        tempFreq[i] = tmp * std::exp((-i_c * M_PIf32 / 4.0f) * sign(vec[i]));
    }

    return prod(tempFreq);
}