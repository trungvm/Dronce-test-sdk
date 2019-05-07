#include "rsCam.h"

RsCamera::RsCamera() {
    if (!initialize_streaming()) {
        std::cerr << "Connect to rs camera: FALL" << std::endl;
        return;
    }
    std::cerr << "Connect to rs camera: SUCCESS" << std::endl;
    process();
}

RsCamera::~RsCamera() {
    _rs_camera->stop( );
}

bool RsCamera::initialize_streaming() {
    bool success = false;
	if( _rs_ctx.get_device_count( ) > 0 )
	{
		_rs_camera = _rs_ctx.get_device( 0 );

		_rs_camera->enable_stream( rs::stream::color, RS_INPUT_WIDTH, RS_INPUT_HEIGHT, rs::format::rgb8, RS_FRAMERATE );
		_rs_camera->enable_stream( rs::stream::depth, RS_INPUT_WIDTH, RS_INPUT_HEIGHT, rs::format::z16, RS_FRAMERATE );

		_rs_camera->start( );

		success = true;
	}
	return success;
}

double RsCamera::findDepthObj(cv::Rect box) {
    double ret = 0;
    int ct = 0, x, y;
    cv::RNG rng(12345);
    for (int i = 0; i < 10; i++) {
        x = rng.uniform(box.tl().x, box.br().x);
        y = rng.uniform(box.tl().y, box.br().y);

        // cv::circle( drawing, cv::Point(x,y), 3, cv::Scalar(255, 0, 0), 2, 8, 0 );

        double d = depMat.at<float>(y, x);
        if (d > (DEPTH_MIN - 1e-6)  && d < (DEPTH_MAX + 1e-6)) {
            ret += d;
            ct++;
        }
    }
    return ret / (ct * 1.f);
    // cv::imshow("tmp", drawing);
    // cv::waitKey();
}

void RsCamera::findBoxObj() {
    cv::findContours( binMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    std::vector< std::vector< cv::Point > > contours_poly( contours.size() );
    // boundRect.clear();
    objPos.clear();
    for( int i = 0; i < contours.size(); i++ ) {
        approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        // boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
        cv::Rect tmp = cv::boundingRect( cv::Mat(contours_poly[i]) );
        if (tmp.height > 50 && tmp.width > 50) {
            // boundRect.push_back(tmp);
            double depth = findDepthObj(tmp);
            objPos.push_back(std::make_pair(tmp, depth));
        }
    }
#ifdef DEBUG_CAM    
    drawing = rgbMat;
    for( int i = 0; i< objPos.size(); i++ ) {
        cv::Scalar color = cv::Scalar( 0, 0, 255 );
        cv::Rect tmp = objPos[i].first;
        double d = objPos[i].second;
        char str[10];
        sprintf(str, "%lf", d);
        // drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
        rectangle( drawing, tmp.tl(), tmp.br(), color, 2, 8, 0 );
        putText(drawing, str, tmp.tl(), cv::FONT_HERSHEY_PLAIN, 2,  cv::Scalar(0,255,0));
        // circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
#endif
}
void RsCamera::process() {

        
        if( _rs_camera->is_streaming( ) )
			_rs_camera->wait_for_frames( );

        double st = cv::getTickCount();
        getFrame();
        findBoxObj();
        double ed = cv::getTickCount();

        std::cerr << 1.0 / ((ed - st) / cv::getTickFrequency()) << std::endl;

    #ifdef DEBUG_CAM
           cv::imshow("rgb", rgbMat);
        if (cv::waitKey(5) == 'q') break;
        cv::imshow("bin", binMat);
        if (cv::waitKey(5) == 'q') break;
        cv::imshow("dep", depMat * 5);
        if (cv::waitKey(5) == 'q') break;
    
        cv::imshow("ct", drawing);
        cv::waitKey(5);
    #endif

        


    return ;

}
void RsCamera::convertToXYZ(cv::Mat & depthMat, rs::intrinsics & camInfo, cv::Mat & xyzMat, cv::Mat & binMat) {
    // const float qnan = std::numeric_limits<float>::quiet_NaN();
    binMat = cv::Mat(RS_INPUT_HEIGHT, RS_INPUT_WIDTH, CV_8UC1);
    // xyzMat = cv::Mat(RS_INPUT_HEIGHT, RS_INPUT_WIDTH, CV_32FC3);
	for (int i = 0; i < depthMat.rows; i++)
	{
		for (int j = 0; j < depthMat.cols; j++)
		{
			float d = depthMat.at<float>(i, j);
			
            // std::cerr << "sample " << d << std::endl;
			if (d < DEPTH_MIN || d > DEPTH_MAX) {
				// xyzMat.at<cv::Point3f>(i, j) = cv::Point3f(qnan, qnan, qnan);
                binMat.at<uchar>(i, j) = 0;
                // depthMat.at<float>(i ,j) = 0.f;
            }
			else {
				// xyzMat.at<cv::Point3f>(i, j) = cv::Point3f((j - (camInfo.width / 2.f - 0.5f)) * d / camInfo.fx, (i - (camInfo.height / 2.f - 0.5f)) * d / camInfo.fy, d);
                binMat.at<uchar>(i, j) = 255;
            }

		}
	}
}
void RsCamera::getFrame() {
    _depth_intrin 	= _rs_camera->get_stream_intrinsics(rs::stream::depth_aligned_to_color);
	_color_intrin 	= _rs_camera->get_stream_intrinsics( rs::stream::color );

    cv::Mat depth16( _depth_intrin.height,
					 _depth_intrin.width,
					 CV_16U,
					 (uchar *)_rs_camera->get_frame_data( rs::stream::depth_aligned_to_color ) );

    depth16.convertTo(depMat, CV_32F);
    depMat /= RS_SCALE;
    cv::Mat rgb( _color_intrin.height,
				 _color_intrin.width,
				 CV_8UC3,
				 (uchar *)_rs_camera->get_frame_data( rs::stream::color ) );
    cv::cvtColor( rgb, rgbMat, cv::COLOR_BGR2RGB );
    
    cv::inRange(depMat, cv::Scalar(DEPTH_MIN), cv::Scalar(DEPTH_MAX), binMat);
    int erosion_size = 5;  
    cv::Mat elementE = getStructuringElement(cv::MORPH_ELLIPSE,
              cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
              cv::Point(erosion_size, erosion_size) );
    cv::erode(binMat, binMat, elementE); 

    int dilation_size = 6;  
    cv::Mat elementD = getStructuringElement(cv::MORPH_ELLIPSE,
              cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
              cv::Point(dilation_size, dilation_size) );
    cv::dilate(binMat, binMat, elementD);
}
