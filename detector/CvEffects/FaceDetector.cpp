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

FaceDetector::FaceDetector()
{
    clnf_model.Read(det_parameters.model_location);
}

void FaceDetector::putImage(Mat& frame, const Mat& image,
                            const Mat& alpha, Rect face,
                            Rect feature, float shift)
{
    // Scale animation image
    float scale = 1.1;
    Size size;
    size.width = scale * feature.width;
    size.height = scale * feature.height;
    Size newSz = Size(size.width,
                      float(image.rows) / image.cols * size.width);
    Mat glasses;
    Mat mask;
    resize(image, glasses, newSz);
    resize(alpha, mask, newSz);

    // Find place for animation
    float coeff = (scale - 1.) / 2.;
    Point origin(face.x + feature.x - coeff * feature.width,
                 face.y + feature.y - coeff * feature.height +
                 newSz.height * shift);
    Rect roi(origin, newSz);
    Mat roi4glass = frame(roi);
    
    alphaBlendC4(glasses, roi4glass, mask);
}

static bool FaceSizeComparer(const Rect& r1, const Rect& r2)
{
    return r1.area() > r2.area();
}

void FaceDetector::PreprocessToGray(Mat& frame)
{
    cvtColor(frame, grayFrame_, CV_RGBA2GRAY);
    equalizeHist(grayFrame_, grayFrame_);
}

void FaceDetector::PreprocessToGray_optimized(Mat& frame)
{
    grayFrame_.create(frame.size(), CV_8UC1);
    accBuffer1_.create(frame.size(), frame.type());
    accBuffer2_.create(frame.size(), CV_8UC1);
        
    cvtColor_Accelerate(frame, grayFrame_, accBuffer1_, accBuffer2_);
    equalizeHist_Accelerate(grayFrame_, grayFrame_);
}

void FaceDetector::detectAndAnimateFaces(Mat& frame)
{
    TS(Preprocessing);
    //PreprocessToGray(frame);
    PreprocessToGray_optimized(frame);
    TE(Preprocessing);
    
    // Detect faces
    TS(DetectFaces);
   
}
