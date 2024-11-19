#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>
#include <cmath>

#include "datatypes.h"

extern const std::complex<float> i_c;
extern const uint16_t maxUInt_16;
extern const float FloatInf;

namespace MathUtils
{   
    // return the symbol of value
    template<class T>
    T sign(T value);

    template<class T>
    T prod(const std::vector<T> &vec);

    std::complex<float> getInitCoeff(const FArray &vec);
    FArray genEquidisRange(float start, float end, int n);

    template<class T>
    std::vector<T> differVector(const std::vector<T> &data);    

    template<class T>
    T diffInnerProduct(const std::vector<T> &vec1, const std::vector<T> &vec2);

    // Apply median filtering to vector
    template <class T>
    std::vector<T> medfilt1(const std::vector<T> &signal, int window_size = 3);  

    /* Calculate the moving average of the vecotr based on the window size,
       and discard data points that cannot fully compute the window */
    template<class T>
    FArray movmean(const std::vector<T>& data, int window_size);

    // Find the L2 norm of two complex vectors
    float complexL2Norm(const ComArray &vec1, const ComArray &vec2);
}

template<class T>
T MathUtils::prod(const std::vector<T> &vec)
{
    return std::accumulate(vec.begin(), vec.end(), static_cast<T>(1), std::multiplies<T>());
}

template<class T>
T MathUtils::sign(T value)
{   
    if (value > 0) return static_cast<T>(1);
    if (value < 0) return static_cast<T>(-1);
    return static_cast<T>(0);
}

template<class T>
std::vector<T> MathUtils::differVector(const std::vector<T> &data)
{
    std::vector<T> result;
    int n = data.size();
    if (n < 2)
        return result;
    
    result.resize(n - 1);
    for (int i = 1; i < n; i++)
    {
        result[i - 1] = data[i] - data[i - 1];
    }

    return result;
}    

template<class T>
T MathUtils::diffInnerProduct(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        throw std::invalid_argument("The lengths of the 2 vectors are not the same!");
    }
    
    std::vector<T> diff(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), diff.begin(), std::minus<T>());
    return std::inner_product(diff.begin(), diff.end(), diff.begin(), static_cast<T>(0));
}

template <class T>
std::vector<T> MathUtils::medfilt1(const std::vector<T> &signal, int window_size) 
{
    if (window_size % 2 == 0 || window_size < 1) {
        throw std::invalid_argument("Window size must be an odd positive integer!");
    }

    int half_window = window_size / 2;
    int signal_size = signal.size();
    std::vector<T> filtered_signal(signal_size);

    for (int i = 0; i < signal_size; ++i) {
        // Define the window range
        int start = std::max(0, i - half_window);
        int end = std::min(signal_size - 1, i + half_window);

        // Extract the window elements
        std::vector<T> window(signal.begin() + start, signal.begin() + end + 1);

        // Find the median
        std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
        filtered_signal[i] = window[window.size() / 2];
    }

    return filtered_signal;
}

template <class T>
FArray MathUtils::movmean(const std::vector<T>& data, int window_size)
{
    FArray result;
    int n = data.size();

    if (window_size > n || window_size <= 0) {
        return result;  // Return empty if window size is invalid
    }

    float sum = std::accumulate(data.begin(), data.begin() + window_size, 0.0f);
    result.push_back(sum / window_size);

    for (int i = window_size; i < n; ++i) {
        sum += data[i] - data[i - window_size];
        result.push_back(sum / window_size);
    }

    return result;
}

#endif