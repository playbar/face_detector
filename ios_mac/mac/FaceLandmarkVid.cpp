
// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "Processing.hpp"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		//double vis_certainty = detection_certainty;
		//if (vis_certainty > 1)
		//	vis_certainty = 1;
		//if (vis_certainty < -1)
		//	vis_certainty = -1;

		//vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		//// A rough heuristic for box around the face width
		//int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		//cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		//// Draw it in reddish if uncertain, blueish if certain
		//LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
		//
		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("facedetector", 1);
		cv::imshow("facedetector", captured_image);

		if (!depth_image.empty())
		{
			// Division needed for visualisation purposes
			imshow("depth", depth_image / 2000.0);
		}

	}
}

void LoadLefEye(LandmarkDetector::CLNF& face_model){
    LandmarkDetector::CLNF left_eye;
    left_eye.Read_CLNF("data/eye/left.dat");
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

void LoadRightEye(LandmarkDetector::CLNF& face_model){
    LandmarkDetector::CLNF right_eye;
    right_eye.Read_CLNF("data/eye/right.dat");
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

void LoadCLNFInner(LandmarkDetector::CLNF& face_model)
{
    LandmarkDetector::CLNF clnf_inner;
    clnf_inner.Read_CLNF("data/inner/clnf_inner.dat");
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


void BuildCLNF(LandmarkDetector::CLNF& face_model, string name )
{
    face_model.Read_CLNF( "data/main.dat" );
   
    LoadLefEye(face_model);
    LoadRightEye(face_model);
    face_model.eye_model = true;
    LoadCLNFInner( face_model);
    //face_model.Read( name );
    face_model.Init();
    face_model.landmark_validator.Read("data/validator.dat");
    
    
}


int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
    vector<string> files;
    vector<string> depth_directories;
    vector<string> output_video_files;
    vector<string> out_dummy;
	
	// By default try webcam 0
	int device = 0;

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to world or camera coordinates
	bool u;
	LandmarkDetector::get_video_input_output_params(files, depth_directories, out_dummy, output_video_files, u, arguments);
	
	// The modules that are being used for tracking
    //LandmarkDetector::CLNF inner("model/model_inner/main_clnf_inner.txt");
    //LandmarkDetector::CLNF left_eye("model/model_eye/main_clnf_synth_left.txt");
    //LandmarkDetector::CLNF right_eye("model/model_eye/main_clnf_synth_right.txt");
    
    LandmarkDetector::CLNF clnf_model;//(det_parameters.model_location);
    BuildCLNF( clnf_model, det_parameters.model_location);

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	
	det_parameters.track_gaze = true;

	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;

		// We might specify multiple video files as arguments
		if(files.size() > 0)
		{
			f_n++;			
		    current_file = files[f_n];
		}
		else
		{
			// If we want to write out from webcam
			f_n = 0;
		}
		
		bool use_depth = !depth_directories.empty();	

		// Do some grabbing
		cv::VideoCapture video_capture;
		if( current_file.size() > 0 )
		{
			if (!boost::filesystem::exists(current_file))
			{
				FATAL_STREAM("File does not exist");
			}

			current_file = boost::filesystem::path(current_file).generic_string();

			INFO_STREAM( "Attempting to read from file: " << current_file );
			video_capture = cv::VideoCapture( current_file );
		}
		else
		{
			INFO_STREAM( "Attempting to capture from device: " << device );
			video_capture = cv::VideoCapture( device );
            video_capture.set(CV_CAP_PROP_FRAME_WIDTH, 600);
            video_capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
            //video_capture.set(CV_CAP_PROP_FPS, 3000);

			// Read a first frame often empty in camera
			cv::Mat captured_image;
			video_capture >> captured_image;
		}

		if( !video_capture.isOpened() )
            FATAL_STREAM( "Failed to open video source" );
		else
            INFO_STREAM( "Device or file opened");

		cv::Mat captured_image;
		video_capture >> captured_image;		

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}		
	
		int frame_count = 0;
		
		// saving the videos
		cv::VideoWriter writerFace;
		if (!output_video_files.empty())
		{
			writerFace = cv::VideoWriter(output_video_files[f_n], CV_FOURCC('D', 'I', 'V', 'X'), 30, captured_image.size(), true);
		}

		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		

			// Reading the images
			cv::Mat_<float> depth_image;
			cv::Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			// Get depth image
			if(use_depth)
			{
				char* dst = new char[100];
				std::stringstream sstream;

				sstream << depth_directories[f_n] << "\\depth%05d.png";
				sprintf(dst, sstream.str().c_str(), frame_count + 1);
				// Reading in 16-bit png image representing depth
				cv::Mat_<short> depth_image_16_bit = cv::imread(string(dst), -1);

				// Convert to a floating point depth image
				if(!depth_image_16_bit.empty())
				{
					depth_image_16_bit.convertTo(depth_image, CV_32F);
				}
				else
				{
					WARN_STREAM( "Can't find depth image" );
				}
			}
			
			// The actual facial landmark detection / tracking
            //bool detection_success = false;
            TS(FaceDetetect);
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);
            TE(FaceDetetect);
			
			// Visualising the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			//double detection_certainty = clnf_model.detection_certainty;

			// Gaze tracking, absolute gaze direction
            
            TS(FaceDraw);
            
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
			{
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
			}

			visualise_tracking(captured_image, depth_image, clnf_model, det_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);
			
           TE(FaceDraw);
            
			// output the tracked video
			if (!output_video_files.empty())
			{
				writerFace << captured_image;
			}


			video_capture >> captured_image;
		
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press == 'r')
			{
				clnf_model.Reset();
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;

		}
		
		frame_count = 0;

		// Reset the model, for the next video
		clnf_model.Reset();
		
		// break out of the loop if done with all the files (or using a webcam)
		if(f_n == files.size() -1 || files.empty())
		{
			done = true;
		}
	}

	return 0;
}

