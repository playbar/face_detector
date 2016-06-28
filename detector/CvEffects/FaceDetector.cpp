/*****************************************************************************
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
    //face_model.Read(det_parameters.model_location);
    BuildCLNF();
}


void FaceDetector::LoadLefEye()
{
    LandmarkDetector::CLNF left_eye;
    string path = det_parameters.strRoot + "/eye/left.dat";
    left_eye.Read_CLNF(path);
    left_eye.Init();
    vector<pair<int, int> > mappings;
    //37 10 38 12 39 14 40 16 41 18
    mappings.push_back( pair<int, int>(36,8));
    mappings.push_back( pair<int, int>(37, 10));
    mappings.push_back( pair<int, int>(38,12));
    mappings.push_back( pair<int, int>(39,14));
    mappings.push_back( pair<int, int>(40,16));
    mappings.push_back( pair<int, int>(41, 18));
    face_model.hierarchical_mapping.push_back(mappings);
    face_model.hierarchical_models.push_back( left_eye );
    face_model.hierarchical_model_names.push_back("left_eye_28");
    LandmarkDetector::FaceModelParameters params;
    params.validate_detections = false;
    params.refine_hierarchical = false;
    params.refine_parameters = false;
    vector<int> windows_large;
    windows_large.push_back(3);
    windows_large.push_back(5);
    windows_large.push_back(9);
    
    vector<int> windows_small;
    windows_small.push_back(3);
    windows_small.push_back(5);
    windows_small.push_back(9);
    
    params.window_sizes_init = windows_large;
    params.window_sizes_small = windows_small;
    params.window_sizes_current = windows_large;
    
    params.reg_factor = 0.5;
    params.sigma = 1.0;
    face_model.hierarchical_params.push_back(params);
}

void FaceDetector::LoadRightEye()
{
    LandmarkDetector::CLNF right_eye;
     string path = det_parameters.strRoot + "/eye/right.dat";
    right_eye.Read_CLNF(path);
    right_eye.Init();
    vector<pair<int, int> > mappings;
    //42 8 43 10 44 12 45 14 46 16 47 18
    mappings.push_back( pair<int, int>(42, 8));
    mappings.push_back( pair<int, int>(43, 10));
    mappings.push_back( pair<int, int>(44, 12));
    mappings.push_back( pair<int, int>(45, 14));
    mappings.push_back( pair<int, int>(46, 16));
    mappings.push_back( pair<int, int>(47, 18));
    face_model.hierarchical_mapping.push_back(mappings);
    face_model.hierarchical_models.push_back( right_eye );
    face_model.hierarchical_model_names.push_back("right_eye_28");
    LandmarkDetector::FaceModelParameters params;
    params.validate_detections = false;
    params.refine_hierarchical = false;
    params.refine_parameters = false;
    vector<int> windows_large;
    windows_large.push_back(3);
    windows_large.push_back(5);
    windows_large.push_back(9);
    
    vector<int> windows_small;
    windows_small.push_back(3);
    windows_small.push_back(5);
    windows_small.push_back(9);
    
    params.window_sizes_init = windows_large;
    params.window_sizes_small = windows_small;
    params.window_sizes_current = windows_large;
    
    params.reg_factor = 0.5;
    params.sigma = 1.0;
    face_model.hierarchical_params.push_back(params);
}

void FaceDetector::LoadCLNFInner()
{
    LandmarkDetector::CLNF clnf_inner;
    string path = det_parameters.strRoot + "/inner/clnf_inner.dat";
    clnf_inner.Read_CLNF(path);
    clnf_inner.Init();
    vector<pair<int, int> > mappings;
    
    mappings.push_back( pair<int, int>(17, 0));
    mappings.push_back( pair<int, int>(18, 1));
    mappings.push_back( pair<int, int>(19, 2));
    mappings.push_back( pair<int, int>(20, 3));
    mappings.push_back( pair<int, int>(21, 4));
    mappings.push_back( pair<int, int>(22, 5));
    mappings.push_back( pair<int, int>(23, 6));
    mappings.push_back( pair<int, int>(24, 7));
    mappings.push_back( pair<int, int>(25, 8));
    mappings.push_back( pair<int, int>(26, 9));
    mappings.push_back( pair<int, int>(27, 10));
    mappings.push_back( pair<int, int>(28, 11));
    mappings.push_back( pair<int, int>(29, 12));
    mappings.push_back( pair<int, int>(30, 13));
    mappings.push_back( pair<int, int>(31, 14));
    mappings.push_back( pair<int, int>(32, 15));
    mappings.push_back( pair<int, int>(33, 16));
    mappings.push_back( pair<int, int>(34, 17));
    mappings.push_back( pair<int, int>(35, 18));
    mappings.push_back( pair<int, int>(36, 19));
    mappings.push_back( pair<int, int>(37, 20));
    mappings.push_back( pair<int, int>(38, 21));
    mappings.push_back( pair<int, int>(39, 22));
    mappings.push_back( pair<int, int>(40, 23));
    mappings.push_back( pair<int, int>(41, 24));
    mappings.push_back( pair<int, int>(42, 25));
    mappings.push_back( pair<int, int>(43, 26));
    mappings.push_back( pair<int, int>(44, 27));
    mappings.push_back( pair<int, int>(45, 28));
    mappings.push_back( pair<int, int>(46, 29));
    mappings.push_back( pair<int, int>(47, 30));
    mappings.push_back( pair<int, int>(48, 31));
    mappings.push_back( pair<int, int>(49, 32));
    mappings.push_back( pair<int, int>(50, 33));
    mappings.push_back( pair<int, int>(51, 34));
    mappings.push_back( pair<int, int>(52, 35));
    mappings.push_back( pair<int, int>(53, 36));
    mappings.push_back( pair<int, int>(54, 37));
    mappings.push_back( pair<int, int>(55, 38));
    mappings.push_back( pair<int, int>(56, 39));
    mappings.push_back( pair<int, int>(57, 40));
    mappings.push_back( pair<int, int>(58, 41));
    mappings.push_back( pair<int, int>(59, 42));
    mappings.push_back( pair<int, int>(60, 43));
    mappings.push_back( pair<int, int>(61, 44));
    mappings.push_back( pair<int, int>(62, 45));
    mappings.push_back( pair<int, int>(63, 46));
    mappings.push_back( pair<int, int>(64, 47));
    mappings.push_back( pair<int, int>(65, 48));
    mappings.push_back( pair<int, int>(66, 49));
    mappings.push_back( pair<int, int>(67, 50));
    
    face_model.hierarchical_mapping.push_back(mappings);
    face_model.hierarchical_models.push_back( clnf_inner );
    face_model.hierarchical_model_names.push_back("inner");
    LandmarkDetector::FaceModelParameters params;
    params.validate_detections = false;
    params.refine_hierarchical = false;
    params.refine_parameters = false;
    vector<int> windows_large;
    windows_large.push_back(9);
    
    vector<int> windows_small;
    windows_small.push_back(9);
    
    params.window_sizes_init = windows_large;
    params.window_sizes_small = windows_small;
    params.window_sizes_current = windows_large;
    
    params.reg_factor = 2.5;
    params.sigma = 1.75;
    params.weight_factor = 2.5;
    face_model.hierarchical_params.push_back(params);
}

void FaceDetector::BuildCLNF( )
{
    string path = det_parameters.strRoot + "/main.dat";
    face_model.Read_CLNF( path );
    
    LoadLefEye();
    LoadRightEye();
    face_model.eye_model = true;
    LoadCLNFInner( );
    face_model.Init();
    path = det_parameters.strRoot + "/validator.dat";
    face_model.landmark_validator.Read(path);
    return;
    
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
    
    
    cvtColor(frame, grayscale_image, CV_BGR2GRAY);
    equalizeHist( grayscale_image, grayscale_image);
    //PreprocessToGray_optimized( frame);

    
    
    detection_success = LandmarkDetector::DetectLandmarksInVideo( grayscale_image, depth_image, face_model, det_parameters );
    
    showDetect(frame);
    

    // Detect faces
   
}

void FaceDetector::showDetect(Mat& frame)
{
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    
    
    if (det_parameters.track_gaze && detection_success && face_model.eye_model)
    {
        FaceAnalysis::EstimateGaze(face_model, gazeDirection0, fx, fy, cx, cy, true);
        FaceAnalysis::EstimateGaze(face_model, gazeDirection1, fx, fy, cx, cy, false);
    }
    
    double detection_certainty = face_model.detection_certainty;
    detection_success = face_model.detection_success;
    
    double visualisation_boundary = 0.2;
    if (detection_certainty < visualisation_boundary)
    {
        LandmarkDetector::Draw(frame, face_model);
        
        if (det_parameters.track_gaze && detection_success && face_model.eye_model)
        {
            FaceAnalysis::DrawGaze(frame, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
        }
    }
    
    //cv::Point center( 100, 100);
    //cv::circle(frame, center, 20, cv::Scalar( 255, 0, 0));
    return;
 
}
