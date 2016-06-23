/*****************************************************************************
 *   FaceAnimator.cpp
 ******************************************************************************
 *   by Kirill Kornyakov and Alexander Shishkov, 13th May 2013
 ****************************************************************************** *
 *   Copyright Packt Publishing 2013.
 *   http://bit.ly/OpenCV_for_iOS_book
 *****************************************************************************/

#include "FaceDetector.hpp"
#include "Processing.hpp"

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

FaceDetector::FaceDetector(vector<string> &arguments)
{
    det_parameters.init();
    det_parameters.initArg(arguments);
    clnf_model.Read(det_parameters.model_location);
}

void FaceDetector::PreprocessToGray_optimized(Mat& frame)
{
    grayscale_image.create(frame.size());
    accBuffer1_.create(frame.size(), frame.type());
    accBuffer2_.create(frame.size(), CV_8UC1);
    
    cvtColor_Accelerate(frame, grayscale_image, accBuffer1_, accBuffer2_);
    equalizeHist_Accelerate(grayscale_image, grayscale_image);
}

void FaceDetector::PreprocessToGray(Mat& frame)
{
    cvtColor(frame, grayFrame_, CV_RGBA2GRAY);
    equalizeHist(grayFrame_, grayFrame_);
}

void FaceDetector::detectAndAnimateFaces(Mat& frame)
{

    
    cx = frame.cols / 2.0f;
    cy = frame.rows / 2.0f;
    
    fx = 500 * (frame.cols / 640.0f);
    fy = 500 * (frame.rows / 480.0f);
    
    fx = (fx + fy) / 2.0;
    fy = fx;
    
    
    TS(Preprocessing);
    cvtColor(frame, grayscale_image, CV_BGR2GRAY);
    equalizeHist( grayscale_image, grayscale_image);
    //PreprocessToGray_optimized( frame);

    TE(Preprocessing);
    
    
    TS(DetectFaces);
    detection_success = LandmarkDetector::DetectLandmarksInVideo( grayscale_image, depth_image, clnf_model, det_parameters );
    //detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, clnf_model, det_parameters);
    
    showDetect(frame);
    
    TE(DetectFaces);

    // Detect faces
   
}

void FaceDetector::showDetect(Mat& frame)
{
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    
    
    if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
    {
        FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
        FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
    }
    
    double detection_certainty = clnf_model.detection_certainty;
    detection_success = clnf_model.detection_success;
    
    double visualisation_boundary = 0.2;
    if (detection_certainty < visualisation_boundary)
    {
        LandmarkDetector::Draw(frame, clnf_model);
        
        if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
        {
            FaceAnalysis::DrawGaze(frame, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
        }
    }
    
    cv::Point center( 100, 100);
    cv::circle(frame, center, 20, cv::Scalar( 255, 0, 0));
    return;
 
}
