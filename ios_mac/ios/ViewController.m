#import "ViewController.h"
#import "LandmarkCoreIncludes.h"
#import "GazeEstimation.h"

@interface ViewController ()

@end

@implementation ViewController

@synthesize imageView;
@synthesize startCaptureButton;
@synthesize toolbar;
@synthesize videoCamera;

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.videoCamera = [[CvVideoCamera alloc]
                        initWithParentView:imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset =AVCaptureSessionPreset352x288;
    self.videoCamera.defaultAVCaptureVideoOrientation =AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
 
    isCapturing = NO;
    
    NSString* filename = [[NSBundle mainBundle]pathForResource:@"data/main" ofType:@"dat"];
    
    vector<string> arguments;
    string strFilename = [filename UTF8String];
    
    arguments.push_back(strFilename);
    faceDetector = new FaceDetector( arguments);
    //LandmarkDetector::FaceModelParameters det_parameters( arguments);
    
    //LandmarkDetector::CLNF clnf_model( det_parameters.model_location);
    //[videoCamera start];
    return;
    
}

- (NSInteger)supportedInterfaceOrientations
{
    // Only portrait orientation
    return UIInterfaceOrientationMaskPortrait;
}

-(IBAction)startCaptureButtonPressed:(id)sender
{
    [videoCamera start];
    isCapturing = YES;
    
}

-(IBAction)stopCaptureButtonPressed:(id)sender
{
    [videoCamera stop];
    isCapturing = NO;
}

// Macros for time measurements
#if 1
  #define TS(name) int64 t_##name = cv::getTickCount()
  #define TE(name) printf("TIMER_" #name ": %.2fms\n", \
    1000.*((cv::getTickCount() - t_##name) / cv::getTickFrequency()))
#else
  #define TS(name)
  #define TE(name)
#endif

- (void)processImage:(cv::Mat&)image
{
//    static int i = 0;
//    if( 2 != i ){
//        i++;
//        faceDetector->showDetect(image);
//        return;
//    }
//    i = 0;
    //TS(DetectAndAnimateFaces);
    //faceAnimator->detectAndAnimateFaces(image);
    faceDetector->detectAndAnimateFaces(image );
    //cv::Point center( 100, 100);
    //cv::circle(image, center, 20, cv::Scalar( 255, 0, 0));
    //TE(DetectAndAnimateFaces);
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)viewDidDisappear:(BOOL)animated
{
    [super viewDidDisappear:animated];
    if (isCapturing)
    {
        [videoCamera stop];
    }
}

- (void)dealloc
{
    videoCamera.delegate = nil;
}

@end
