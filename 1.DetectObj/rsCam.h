#include <librealsense/rs.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <algorithm>

#define DEBUG_CAM

int const RS_INPUT_WIDTH 	= 320;
int const RS_INPUT_HEIGHT 	= 240;
int const RS_FRAMERATE 	= 60;
double const RS_SCALE = 1000.f; // convert mm -> m
double const DEPTH_MIN = 0.5f;
double const DEPTH_MAX = 1.5f;

class RsCamera {
    public:
        RsCamera();
        ~RsCamera();
        void process();
        std::vector< std::pair <cv::Rect, double> > objPos;
    private:
        bool initialize_streaming();
        void getFrame();
        double findDepthObj(cv::Rect box);
        void convertToXYZ(cv::Mat & depthMat, rs::intrinsics & camInfo, cv::Mat & xyzMat, cv::Mat & binMat);
        void findBoxObj();

        rs::context 	_rs_ctx;
        rs::device* 	_rs_camera = NULL;
        rs::intrinsics 	_depth_intrin;
        rs::intrinsics  _color_intrin;

        cv::Mat rgbMat, depMat, xyzMat, binMat, drawing;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        std::vector< cv::Rect> boundRect;

};
