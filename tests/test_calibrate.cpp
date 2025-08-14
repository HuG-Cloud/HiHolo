#include <iostream>
#include <string>
#include <cmath>
#include "image_utils.h"

const int numImages = 4;

int main()
{
    const std::string inputPath = "/home/hug/Downloads/HoloTomo_Data/result_";
    DArray nz {0.3, 0.5, 0.7, 0.9};
    
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> profiles(numImages);
    std::vector<cv::Mat> frequencies(numImages);
    
    // Load and convert images to grayscale
    for (int i = 0; i < numImages; i++)
    {
        cv::Mat colorImage = cv::imread(inputPath + std::to_string(i) + ".tif", cv::IMREAD_UNCHANGED);
        if (colorImage.empty()) {
            std::cerr << "Cannot read image: " << inputPath + std::to_string(i) + ".tif" << std::endl;
            continue;
        }

        cv::Mat grayImage;
        if (colorImage.channels() > 1) {
            cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
        } else {
            grayImage = colorImage.clone();
        }

        cv::Mat grayImg32F;
        grayImage.convertTo(grayImg32F, CV_32F);
        images.push_back(grayImg32F);
    }

    // Compute power spectral density for all images
    int direction = 0; // Compute PSD along columns
    DArray maxPSD = ImageUtils::computePSDs(images, direction, profiles, frequencies);
    
    // Print maximum PSD values
    for (int i = 0; i < numImages; i++) {
        std::cout << "Image " << i << " (n=" << nz[i] << ") max PSD: " << maxPSD[i] << std::endl;
    }
    
    // Plot parameters
    const int plotHeight = 600;
    const int plotWidth = 1000;
    const int margin = 80;
    
    cv::Mat plotImage(plotHeight, plotWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 绘制坐标轴
    cv::line(plotImage, cv::Point(margin, plotHeight - margin), 
             cv::Point(plotWidth - margin, plotHeight - margin), cv::Scalar(0, 0, 0), 2); // X轴
    cv::line(plotImage, cv::Point(margin, margin), 
             cv::Point(margin, plotHeight - margin), cv::Scalar(0, 0, 0), 2); // Y轴
    
    // 只绘制第一个图像的频谱曲线
    if (!profiles[0].empty() && !frequencies[0].empty()) {
        // 找到第一个图像的数据范围
        double minFreq, maxFreq, minProf, maxProf;
        cv::minMaxLoc(frequencies[0], &minFreq, &maxFreq);
        cv::minMaxLoc(profiles[0], &minProf, &maxProf);
        
        double freqRange = maxFreq - minFreq;
        double profRange = maxProf - minProf;
        if (freqRange == 0) freqRange = 1.0;
        if (profRange == 0) profRange = 1.0;
        
        std::vector<cv::Point> points;
        
        // 将频率和功率数据转换为绘图坐标
        float* freqData = (float*)frequencies[0].data;
        float* profData = (float*)profiles[0].data;
        int totalPoints = static_cast<int>(frequencies[0].total());
        
        for (int j = 0; j < totalPoints; j++) {
            int x = margin + static_cast<int>((plotWidth - 2*margin) * (freqData[j] - minFreq) / freqRange);
            int y = (plotHeight - margin) - static_cast<int>((plotHeight - 2*margin) * (profData[j] - minProf) / profRange);
            points.push_back(cv::Point(x, y));
        }
        
        // 绘制曲线（使用蓝色）
        cv::Scalar color = cv::Scalar(255, 0, 0); // 蓝色
        for (size_t j = 1; j < points.size(); j++) {
            cv::line(plotImage, points[j-1], points[j], color, 2);
        }
        
        // 添加图例
        cv::line(plotImage, cv::Point(plotWidth - 200, 50), 
                 cv::Point(plotWidth - 170, 50), color, 3);
        std::string label = "n=" + std::to_string(nz[0]).substr(0, 3);
        cv::putText(plotImage, label, cv::Point(plotWidth - 165, 55), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        
        // 更新坐标轴刻度使用第一个图像的数据范围
        // X轴刻度
        const int numXTicks = 6;
        for (int i = 0; i <= numXTicks; i++) {
            double freqValue = minFreq + (double)i / numXTicks * freqRange;
            int x = margin + (plotWidth - 2*margin) * i / numXTicks;
            
            cv::line(plotImage, cv::Point(x, plotHeight - margin), cv::Point(x, plotHeight - margin + 5), cv::Scalar(0, 0, 0), 1);
            
            std::string tickLabel = std::to_string(freqValue).substr(0, 4);
            cv::putText(plotImage, tickLabel, cv::Point(x - 15, plotHeight - margin + 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
        
        // Y轴刻度
        const int numYTicks = 5;
        for (int i = 0; i <= numYTicks; i++) {
            double profValue = minProf + (double)i / numYTicks * profRange;
            int y = (plotHeight - margin) - (plotHeight - 2*margin) * i / numYTicks;
            
            cv::line(plotImage, cv::Point(margin - 5, y), cv::Point(margin, y), cv::Scalar(0, 0, 0), 1);
            
            std::string tickLabel = std::to_string(profValue).substr(0, 6);
            cv::putText(plotImage, tickLabel, cv::Point(margin - 60, y + 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
    }
    
    // 添加标题和轴标签
    cv::putText(plotImage, "Power Spectral Density", cv::Point(plotWidth / 2 - 120, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
                
    // Display and save plot
    cv::namedWindow("Power Spectral Density", cv::WINDOW_AUTOSIZE);
    cv::imshow("Power Spectral Density", plotImage);
    cv::waitKey(0);
    
    cv::imwrite("power_spectrum_density.png", plotImage);
    std::cout << "功率频谱图已保存为 power_spectrum_density.png" << std::endl;

    return 0;
}