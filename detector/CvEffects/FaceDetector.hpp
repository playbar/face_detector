/*****************************************************************************
 *   FaceDetector.hpp
 ******************************************************************************
 *   by Kirill Kornyakov and Alexander Shishkov, 13th May 2013
 ****************************************************************************** *
 *   Copyright Packt Publishing 2013.
 *   http://bit.ly/OpenCV_for_iOS_book
 *****************************************************************************/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"

class FaceDetector
{
public:

    FaceDetector(vector<string> &arguments);
    virtual ~FaceDetector() {};

    void detectAndAnimateFaces(cv::Mat& frame);
    
    void PreprocessToGray_optimized(cv::Mat& frame);
    void PreprocessToGray(cv::Mat& frame);

protected:
    
    LandmarkDetector::FaceModelParameters det_parameters;
    LandmarkDetector::CLNF clnf_model;
    
    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;
    
    cv::Mat maskOrig_;
    cv::Mat maskMust_;
    cv::Mat grayFrame_;
    cv::Mat accBuffer1_;
    cv::Mat accBuffer2_;

};
