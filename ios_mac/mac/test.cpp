
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
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image,
    const LandmarkDetector::FaceModelParameters& det_parameters,
    cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count,
    double fx, double fy, double cx, double cy)
{

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
		cv::namedWindow("test", 1);
		cv::imshow("test", captured_image);

		if (!depth_image.empty())
		{
			// Division needed for visualisation purposes
			imshow("depth", depth_image / 2000.0);
		}

	}
}

void TestImg()
{
    IplImage *img1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1 );
    IplImage *img2;
    img2 = cvCloneImage(img1 );
    return;
}

void cvShiftDFT( CvArr *src_arr, CvArr *dst_arr){
    CvMat *tmp;
    CvMat q1stub, q2stub;
    CvMat q3stub, q4stub;
    CvMat d1stub, d2stub;
    CvMat d3stub, d4stub;
    CvMat *q1, *q2, *q3, *q4;
    CvMat *d1, *d2, *d3, *d4;
    CvSize size = cvGetSize(src_arr);
    CvSize dst_size = cvGetSize(dst_arr );
    
    int cx, cy;
    
    if( dst_size.width != size.width || dst_size.height != size.height ){
        cvError( CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__);
    }
    
    if( src_arr == dst_arr ){
        tmp = cvCreateMat( size.height / 2, size.width / 2, cvGetElemType( src_arr ));
    }
    
    cx = size.width / 2;
    cy = size.height / 2;
    
    q1 = cvGetSubRect(src_arr, &q1stub, cvRect(0, 0, cx, cy ));
    q2 = cvGetSubRect(src_arr, &q2stub, cvRect(cx, 0, cx, cy));
    q3 = cvGetSubRect(src_arr, &q3stub, cvRect(cx, cy, cx, cy));
    q4 = cvGetSubRect(src_arr, &q4stub, cvRect(0, cy, cx, cy));
    d1 = cvGetSubRect(src_arr, &d1stub, cvRect(0, 0, cx, cy));
    d2 = cvGetSubRect(src_arr, &d2stub, cvRect(cx, 0, cx, cy));
    d3 = cvGetSubRect(src_arr, &d3stub, cvRect(cx, cy, cx, cy));
    d4 = cvGetSubRect(src_arr, &d4stub, cvRect(0, cy, cx, cy));
    
    if( src_arr != dst_arr ){
        if( CV_ARE_TYPES_EQ(q1, d1)){
            cvError(CV_StsUnmatchedFormats, "cvShiftDFT", "Source and Destination", __FILE__, __LINE__ );
        }
        cvCopy(q3, d1, 0 );
        cvCopy(q4, d2, 0 );
        cvCopy(q1, d3, 0 );
        cvCopy(q2, d4, 0);
    }
    else{
        cvCopy(q3, tmp,  0);
        cvCopy(q1, q3, 0);
        cvCopy( tmp, q1, 0);
        cvCopy(q4, tmp, 0);
        cvCopy(q2, q4, 0);
        cvCopy(tmp, q2, 0);
    }
    return;
    
    
}

int main (int argc, char **argv)
{

    TestImg();
    
	vector<string> arguments = get_arguments(argc, argv);
    
    int height, width, step, channels;
    uchar *data;

	LandmarkDetector::FaceModelParameters det_parameters(arguments);
    std::string filename = det_parameters.strRoot + "/test.jpg";
    IplImage *img = cvLoadImage(filename.c_str());
    
    IplImage *im;
    IplImage *realInput;
    IplImage *imaginaryInput;
    IplImage *complexInput;
    int dft_M, dft_N;
    CvMat *dft_A, tmp;
    IplImage *image_Re;
    IplImage *image_Im;
    double m, M;
    im = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE );
    if( !im )
        return -1;
    
    realInput = cvCreateImage(cvGetSize(im), IPL_DEPTH_64F, 1);
    imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1 );
    complexInput = cvCreateImage(cvGetSize(im), IPL_DEPTH_64F, 2);
    
    cvScale( im, realInput, 1.0, 0.0);
    cvZero( imaginaryInput );
    cvMerge( realInput, imaginaryInput, NULL, NULL, complexInput);
    
    dft_M = cvGetOptimalDFTSize( im->height - 1 );
    dft_N = cvGetOptimalDFTSize( im->width - 1 );
    dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2);
    
    image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1 );
    image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1 );
    
    cvGetSubRect( dft_A, &tmp, cvRect( 0, 0, im->width, im->height ));
    cvCopy( complexInput, &tmp, NULL );
    cvGetSubRect( dft_A, &tmp, cvRect(im->width, 0, dft_A->cols - im->width, im->height));
    cvZero( &tmp );
    
    cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );
    
    cvNamedWindow("win", 0);
    cvNamedWindow("magnitude", 0);
    cvShowImage("win", im);
    
    cvSplit( dft_A, image_Re, image_Im, 0, 0);
    
    cvPow( image_Re, image_Re, 2.0 );
    cvPow( image_Im, image_Im, 2.0 );
    cvAdd( image_Re, image_Im, image_Re, NULL );
    cvPow( image_Re, image_Re, 0.5);
    
    cvAddS(image_Re, cvScalarAll(1.0), image_Re, NULL );
    cvLog( image_Re, image_Re );
    cvShiftDFT( image_Re, image_Re );
    cvMinMaxLoc( image_Re, &m, &M, NULL, NULL, NULL);
    cvScale( image_Re, image_Re, 1.0 / (M - m), 1.0 * ( -m) / (M - m));
    cvShowImage("magnitude", image_Re );
    cvWaitKey( -1);
    return 0;
    
}

