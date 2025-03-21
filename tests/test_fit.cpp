#include <iostream>
#include <string>
#include <gsl/gsl_fit.h>


int main()
{
    // 准备带有一些误差的数据点
    const int n = 10;  // 数据点个数
    double x[n], y[n];
    
    // 生成带有随机误差的数据点 (y = 2x + 1)
    for(int i = 0; i < n; i++) {
        x[i] = i;
        // 加入±0.2范围内的随机误差
        double error = -0.2 + (rand() % 100) / 250.0;
        y[i] = 2 * x[i] + 1 + error;
    }

    double c0, c1, cov00, cov01, cov11, sumsq;

    // 进行线性拟合 y = c1*x + c0
    gsl_fit_linear(x, 1, y, 1, n, 
                  &c0,    // y截距
                  &c1,    // 斜率
                  &cov00, // 协方差矩阵元素
                  &cov01, 
                  &cov11,
                  &sumsq  // 残差平方和
    );

    std::cout << "拟合结果:" << std::endl;
    std::cout << "y = " << c1 << "x + " << c0 << std::endl;
    std::cout << "协方差矩阵:" << std::endl;
    std::cout << cov00 << " " << cov01 << std::endl;
    std::cout << cov01 << " " << cov11 << std::endl;
    std::cout << "残差平方和: " << sumsq << std::endl;
}
    