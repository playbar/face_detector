/*****************************************************************************
 *****************************************************************************/

#pragma once

#include <opencv2/core/core.hpp>

void alphaBlendC1(const cv::Mat& src, cv::Mat& dst,
                  const cv::Mat& alpha);
void alphaBlendC4(const cv::Mat& src, cv::Mat& dst,
                  const cv::Mat& alpha);

void ExtractAlpha(cv::Mat& rgbaSrc, cv::Mat& alpha);

// NEON-optimized functions
void alphaBlendC1_NEON(const cv::Mat& src, cv::Mat& dst,
                       const cv::Mat& alpha);
void multiply_NEON(cv::Mat& src, float multiplier);

// Accelerate-optimized functions
int cvtColor_Accelerate(const cv::Mat& src, cv::Mat& dst,
                        cv::Mat buff1, cv::Mat buff2);

int equalizeHist_Accelerate(const cv::Mat& src, cv::Mat& dst);

// Macros for time measurements
#if 1
  #define TS(name) int64 t_##name = cv::getTickCount()
  #define TE(name) printf("TIMER_" #name ": %.2fms\n", \
    1000.f * ((cv::getTickCount() - t_##name) / cv::getTickFrequency()))
#else
  #define TS(name)
  #define TE(name)
#endif
