//
//  CameraCalibration.mm
//  LaneDetection
//
//  Created by charles zeng on 6/15/24.
//
//  references
//   1. https://github.com/ndrplz/self-driving-car.git
//   2. https://github.com/kipr/opencv/blob/master/modules/contrib/src/polyfit.cpp

#import <Foundation/Foundation.h>
#include "CameraCalibration.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <numeric>

void CameraCalibrationData::warpPerspective(const cv::Mat& source, cv::Mat& target, bool isReverse) const
{
    cv::Size sz;
    cv::Mat v;
    if (isReverse)
    {
        cv::warpPerspective(source, target, reverseWarpMatrix, sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    }
    else
    cv::warpPerspective(source, target, warpMatrix, sz, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}
CameraCalibrationData::CameraCalibrationData(const string& plistFile) noexcept(false)
{
    NSString *plist = [NSString stringWithCString:plistFile.c_str()
                                       encoding:[NSString defaultCStringEncoding]];
    if (plist == NULL)  throw 2;
    NSDictionary *theDict = [NSDictionary dictionaryWithContentsOfFile:plist];
    NSDictionary* intrinsics =[theDict objectForKey:@"Intrinsics"];
    isPinhole=[[intrinsics objectForKey:@"isPinhole"] boolValue];
    if (!isPinhole)
        throw 1;
    generatePerspectiveMatrix();

}

shared_ptr<CameraCalibrationData> CameraCalibrationData::getCameraCalibrationData() noexcept(false)
{
    const string calibrationPlist = "CameraCalibration.plist";
    
    shared_ptr<CameraCalibrationData> cameraCalibration;
    cameraCalibration = make_shared<CameraCalibrationData>(calibrationPlist);
    return cameraCalibration;
    
    
}

void CameraCalibrationData::generatePerspectiveMatrix()
{
    // vertices were manually picked from a provided image
    const static cv::Point2f sourceVertices[4]  =
    {
        {1703, 845}, //br
        {533, 845},  //bl
        {854, 495},  //tl
        {1185, 495}  //tr
    };
    const  static cv::Point2f destinationVertices[4] =
    {
        {1703, 845},
        {533, 845},
        {533, 495},
        {1703, 495}
    };
    warpMatrix = cv::getPerspectiveTransform(sourceVertices, destinationVertices);
    reverseWarpMatrix = cv::getPerspectiveTransform(destinationVertices, sourceVertices);

}



bool Binarize::HSVFilter(const cv::Mat& frameRGB, cv::Mat& frameMask)
{
    cv::Mat frameHSV;
    cv::cvtColor(frameRGB, frameHSV, cv::COLOR_RGB2HSV);
    cv::inRange(frameHSV, cv::Scalar(yellow_HSV_min[0], yellow_HSV_min[1], yellow_HSV_min[2]), cv::Scalar(yellow_HSV_max[0], yellow_HSV_max[1], yellow_HSV_max[2]), frameMask);
    return true;
}

bool Binarize::EqualizedFilter(const cv::Mat& frame, cv::Mat& frameEqualized)
{
    cv::Mat gray;
    cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    cv::Mat hist;
    equalizeHist(gray, hist);
    cv::threshold( hist, frameEqualized, 250, 255, cv::THRESH_BINARY);
    return true;
}
bool Binarize::SobelFilter(const cv::Mat& frameRGB, cv::Mat& frameSobelMask)
{
    const int ksize = 9;
    const int scale = 1;
    const int delta = 0;
    const double threshold_value = 50;
    int ddepth = CV_64F;
    cv::Mat frameBlurred;
    int w, h;
   
    GaussianBlur(frameRGB, frameBlurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat frameGray;
    cvtColor(frameBlurred, frameGray, cv::COLOR_RGB2GRAY);
    auto sz = frameGray.size();
    w = sz.width;
    h = sz.height;
    cv::Mat frameGradX(h,w, CV_64F), frameGradY(w, h, CV_64F);
    cv::Sobel(frameGray, frameGradX, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(frameGray, frameGradY, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Mat frameCombined(h, w, CV_64F);
    cv::Mat gradX, gradY;
    frameGradX.convertTo(gradX, CV_64F);
    frameGradY.convertTo(gradY, CV_64F);
    double maxValue = -1;
    for (int i = 0; i < gradX.rows; ++i)
    {
        for (int j = 0; j < gradX.cols; ++j)
        {
            frameCombined.at<double>(i, j) = frameGradX.at<double>(i, j) * frameGradX.at<double>(i, j) + frameGradY.at<double>(i, j) * frameGradY.at<double>(i, j);
            frameCombined.at<double>(i, j) = sqrt(frameCombined.at<double>(i, j));
            if (maxValue < frameCombined.at<double>(i, j))
                maxValue = frameCombined.at<double>(i, j);
            
        }
    }
    cv::Mat frame8U(h, w, CV_8U);
    for (int i = 0; i < gradX.rows; ++i)
    {
        for (int j = 0; j < gradX.cols; ++j)
        {
            frame8U.at<char>(i, j) = (char)(frameCombined.at<double>(i, j)/maxValue * 255.0);
        }
    }
    // maxValue is arbitrarily selected for debugging purpose
    cv::threshold(frame8U, frameSobelMask, threshold_value, 128, cv::THRESH_BINARY);
#if DEBUG
    show(frame8U, "Frame8U");
#endif
    return true;
}

bool Binarize::process(const cv::Mat& frameRGB, cv::Mat& frameOut)
{
    if (frameRGB.channels() != 3)
        return false;
    cv::Mat frameHSVMask, frameSobelMask;
    HSVFilter(frameRGB, frameHSVMask);
    SobelFilter(frameRGB, frameSobelMask);
    cv::Mat equalizedMask;
    EqualizedFilter(frameRGB, equalizedMask);
    cv::Mat mask;
    cv::bitwise_or(equalizedMask, frameHSVMask, mask);
    cv::Mat mask2;
    cv::Mat frameBinary;
    cv::bitwise_or(mask, frameSobelMask, frameBinary);
    MorphFilter(frameBinary, mask2);
    
    // maxValue is arbitrarily selected for debugging purpose
    cv::threshold( mask2, frameOut, 50, 255, cv::THRESH_BINARY);
    return true;
}

bool Binarize::MorphFilter(const cv::Mat& frameBinary, cv::Mat& frameMorphed)
{
    const int morph_size = 2; // image may be blurry if too big
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ));
    morphologyEx( frameBinary, frameMorphed, cv::MORPH_CLOSE, element);
    return true;
}
bool Binarize::HistogramFilter(const cv::Mat& frameRGB, cv::Mat& frameHist)
{
    const int threshold = 250; // arbitrarily selected for debugging purpose
    const int maxVal = 255; // arbitrarily selected for debugging purpose
    cv::Mat frameGray;
    cvtColor(frameRGB, frameGray, cv::COLOR_RGB2GRAY);
    cv::Mat hist;
    cv::equalizeHist( frameGray, hist);
    cv::threshold( hist, frameHist, threshold, maxVal, cv::THRESH_BINARY);
    return true;
}
/*
    minor change from https://github.com/kipr/opencv/blob/master/modules/contrib/src/polyfit.cpp
 */
void DetectAndMarkLane::polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
            &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
    cv::Mat X;
    X = cv::Mat::zeros(src_x.rows, order+1,CV_32FC1);
    cv::Mat copy;
    for(int i = 0; i <=order;i++)
    {
        copy = src_x.clone();
        pow(copy,i,copy);
        cv::Mat M1 = X.col(i);
        copy.col(0).copyTo(M1);
    }
    cv::Mat X_t, X_inv;
    cv::transpose(X,X_t);
    cv::Mat temp = X_t*X;
    cv::Mat temp2;
    
    cv::invert (temp,temp2);
    cv::Mat temp3 = temp2*X_t;
    cv::Mat W = temp3*src_y;
    for (int i = 0; i <= order; ++i)
        dst.at<float>(0, i) = W.at<float>(0, order - i);
    //W.copyTo(dst);
}

bool DetectAndMarkLane::detectLane (cv::Mat &frameBinary, cv::Mat&frameOut, cv::Vec4i& leftLine, cv::Vec4i& rightLine)
{
    try {
        int n_windows = 9; // hardcoded
        auto sz = frameBinary.size();
        int height = sz.height;
        int width = sz.width;
        
        cv::Mat croppedImage;
        frameBinary(cv::Rect(0,height/2,width,height/2)).copyTo(croppedImage);
        std::vector<int> histogram(frameBinary.size().width);

        int leftx_max = 0, rightx_max = 0;
        int leftx_base = 0, rightx_base = width/2;
        
        std::vector<pair<int, int>> nonzero;
        std::vector<int> nonzero_x;
        std::vector<int> nonzero_y;
        
        for (int i = height/2; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if (frameBinary.at<unsigned char>(i, j) > 0)
                    ++histogram[j];
            }
        }
        for (int i = 0; i < width/2; ++i)
        {
            if (histogram[i] > leftx_max)
            {
                leftx_max = histogram[i];
                leftx_base = i;
            }
            if (histogram[i + width/2] > rightx_max)
            {
                rightx_max = histogram[i + width/2];
                rightx_base = i + width/2;
            }
        }
        
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if (frameBinary.at<unsigned char>(i, j) > 0)
                {
                    nonzero.push_back({i, j});
                    nonzero_x.push_back(j);
                    nonzero_y.push_back(i);
                }
            }
        }
        
        cv::Mat out_img;
        std::vector<cv::Mat> vChannels;
        for (unsigned int c = 0; c < 3; c++)
        {
             vChannels.push_back(frameBinary);
        }
        cv::merge(vChannels.data(), 3, out_img);
        std::vector<float> left_lane_inds_x, left_lane_inds_y;
        std::vector<float> right_lane_inds_x, right_lane_inds_y;
        
        int window_height = height/n_windows;
        int margin = 100;  // width of the windows +/- margin
        int minpix = 50;  // minimum number of pixels found to recenter window
        int leftx_current = leftx_base;
        int rightx_current = rightx_base;

        for (int window = 0; window < n_windows; ++window)
        {
            int win_y_low = height - (window + 1) * window_height;
            int win_y_high = height - (window * window_height);
            int win_xleft_low = leftx_current - margin;
            int win_xleft_high = leftx_current + margin;
            int win_xright_low = rightx_current - margin;
            int win_xright_high = rightx_current + margin;
            cv::rectangle(out_img, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(0, 255, 0),2);
            cv::rectangle(out_img, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(0, 255, 0),2);
#if DEBUG
            show(out_img, string("drawing window") + to_string(window));
#endif
            std::vector<float> good_left_inds_x, good_right_inds_x;
            std::vector<float> good_left_inds_y, good_right_inds_y;
            for (int i = 0; i < nonzero.size(); ++i)
            {
                if (nonzero_x[i] >= win_xleft_low && nonzero_x[i] <= win_xleft_high &&
                    nonzero_y[i] >= win_y_low && nonzero_y[i] <= win_y_high)
                {
                    good_left_inds_x.push_back(nonzero_x[i]);
                    good_left_inds_y.push_back(nonzero_y[i]);
                    left_lane_inds_x.push_back(nonzero_x[i]);
                    left_lane_inds_y.push_back(nonzero_y[i]);
                }
                if (nonzero_x[i] >= win_xright_low && nonzero_x[i] <= win_xright_high &&
                    nonzero_y[i] >= win_y_low && nonzero_y[i] <= win_y_high)
                {
                    good_right_inds_x.push_back(nonzero_x[i]);
                    good_right_inds_y.push_back(nonzero_y[i]);
                    right_lane_inds_x.push_back(nonzero_x[i]);
                    right_lane_inds_y.push_back(nonzero_y[i]);
                }
            }
            if (good_left_inds_x.size() > minpix)
            {
                double cnt = 0;
                cnt = std::accumulate(good_left_inds_x.begin(), good_left_inds_x.end(), 0);
                cnt /= good_left_inds_x.size();
                
                leftx_current = cnt;
                
            }
            if (good_right_inds_x.size() > minpix)
            {
                double cnt = 0;
                cnt = std::accumulate(good_right_inds_x.begin(), good_right_inds_x.end(), 0);
                cnt /= good_right_inds_x.size();
                rightx_current = cnt;
                
            }
            std::cout << "leftx_current: " << leftx_current << ", rightx_current: " << rightx_current << endl;
        }
        cv::Mat left_x(cv::Size(1,  (int)left_lane_inds_x.size()), CV_32FC1);
        memcpy(left_x.data,left_lane_inds_x.data(),left_lane_inds_x.size()*sizeof(float));
        cv::Mat left_y(cv::Size(1,  (int)left_lane_inds_y.size()), CV_32FC1);
        memcpy(left_y.data,left_lane_inds_y.data(), (int)left_lane_inds_y.size()*sizeof(float));
        
        
        cv::Mat right_x(cv::Size(1, (int)right_lane_inds_x.size()), CV_32FC1);
        memcpy(right_x.data,right_lane_inds_x.data(),right_lane_inds_x.size()*sizeof(float));
        cv::Mat right_y(cv::Size(1,  (int)right_lane_inds_y.size()), CV_32FC1);
        memcpy(right_y.data,right_lane_inds_y.data(), (int)right_lane_inds_y.size()*sizeof(float));
        
        cv::Mat leftFit(3, 1, CV_32FC1), rightFit(3, 1, CV_32FC1);
        leftFit.at<float>(0, 2) = rightFit.at<float>(0, 2) = 1.0;
        if (left_y.size().width > 0)
            polyfit(left_y, left_x, leftFit, 2);
        if (right_x.size().width > 0)
            polyfit(right_y, right_x, rightFit, 2);
        bool detected = !(left_x.empty() || right_x.empty());
        if (detected)
        {
            _lastFitPixLeft = leftFit;
            _lastFitPixelRight = rightFit;
            _recentFitPixelLeft.push_back(leftFit);
            _recentFitPixelRight.push_back(rightFit);
            if (_recentFitPixelLeft.size() > 30)
                _recentFitPixelLeft.pop_front();
            if (_recentFitPixelRight.size() > 30)
                _recentFitPixelRight.pop_front();
        }
    }catch(...)
    {
        return false;
    }

    return true;
    
}
void DetectAndMarkLane::setMatrixZero(cv::Mat& m) const
{
    int w = m.cols;
    int h = m.rows;
    cv::Rect rectZero(0, 0, w, h);
    m(rectZero) = cv::Scalar(0,0,0);
}
void DetectAndMarkLane::drawWarpedLine(cv::Mat& warped, std::vector<cv::Point>& points)
{
    std::vector<cv::Point> vertice;
    for (int i = 0; i < points.size(); ++i)
    {
        int x = points[i].x, y = points[i].y;
        vertice.emplace_back(x - 25, y);
        vertice.emplace_back(x+ 25, y);
    }
    cv::fillPoly(warped, vertice, cv::Scalar(255, 0, 0), cv::LINE_8);
}
bool DetectAndMarkLane::markLane(const cv::Mat& rawImage, cv::Mat& blendedImage)
{
    std::vector<cv::Point> leftPoints, rightPoints;
    auto sz = rawImage.size();
    std::vector<cv::Point> points;
    int width = sz.width, height = sz.height;
    for (int h = 0; h < height; ++h)
    {
        int  lx = _lastFitPixLeft.at<float>(0, 0)* h * h + _lastFitPixLeft.at<float>(0, 1)* h + _lastFitPixLeft.at<float>(0, 2);
        int  rx = _lastFitPixelRight.at<float>(0, 0)* h * h + _lastFitPixelRight.at<float>(0, 1)* h + _lastFitPixelRight.at<float>(0, 2);
        leftPoints.emplace_back(lx, h);
        rightPoints.emplace_back(rx, h);
        points.emplace_back(lx, h);
        points.emplace_back(rx, h);
    }
    cv::Mat roadWarped, roadUnwarped;
    roadWarped = cv::Mat(height, width, CV_8UC3);
    setMatrixZero(roadWarped);
    cv::fillPoly(roadWarped, points, cv::Scalar(0, 255, 0), cv::LINE_8);
    if (pipeline.use_count() == 0) return false;
    auto PerspectiveWarp = pipeline.lock()->getStage("PerspectiveWarp");
    PerspectiveWarp->setReverseWarp(true);
    PerspectiveWarp->process(roadWarped, roadUnwarped);
    cv::Mat imgWithFilledRoad;
    cv::addWeighted( rawImage, 1.0, roadUnwarped, 0.3, 0.0, imgWithFilledRoad);
    cv::Mat lineWarped, lineUnwarped;
    lineWarped = cv::Mat(height, width, CV_8UC3);
    setMatrixZero(lineWarped);
    drawWarpedLine(lineWarped, leftPoints);
    drawWarpedLine(lineWarped, rightPoints);
    PerspectiveWarp->process(lineWarped, lineUnwarped);
    
    cv::addWeighted( imgWithFilledRoad, 0.8, lineUnwarped, 0.5, 0.0, blendedImage);
#if DEBUG
    {
        show(blendedImage, "blendedImage");
    }
#endif
    return true;
}

LaneDetectionPipeline::LaneDetectionPipeline()
{
    
}
shared_ptr<PipelineStage> LaneDetectionPipeline::getStage(const std::string& name)
{
    if (stages.count(name) == 0) return {};
    return stages[name];
}

bool LaneDetectionPipeline::createPipeline() noexcept
{
    try {
        camera = CameraCalibrationData::getCameraCalibrationData();
    }
    catch(...)
    {
        cerr<< " failed to read camera calibration data " << endl;
        return false;
    }
    auto undistortStage = make_shared<Undistort>(camera);
    stages[undistortStage->getName()] = undistortStage;
    auto binarizedStage = make_shared<Binarize>(undistortStage);
    stages[binarizedStage->getName()] = binarizedStage;
    undistortStage->setNextStage(binarizedStage);
    auto detectAndMarkLaneStage = make_shared<DetectAndMarkLane>(binarizedStage);
    stages[detectAndMarkLaneStage->getName()] = detectAndMarkLaneStage;
    binarizedStage->setNextStage(detectAndMarkLaneStage);
    auto perspectiveWarpStage = make_shared<PerspectiveWarp>(camera);
    stages[perspectiveWarpStage->getName()] = perspectiveWarpStage;
    return true;
}

bool LaneDetectionPipeline::processOneFrame(const cv::Mat& img)
{
    assert(stages.size() >= 3);
    assert(img.size().width && img.size().height);
    /*
         Step One: undistort image if necessary
     */
    auto undistortStage = getStage("Undistort");
    assert(undistortStage);
    cv::Mat imgUndistorted;
    undistortStage->process(img, imgUndistorted);
    /*
         Step Two:
         Apply Gaussian/HSV/Sobel/Equalize/Morph filter,
         then threshold.
     */
    auto binarizeStage = getStage("Binarize");
    assert(binarizeStage);
    cv::Mat imgBinarized;
    binarizeStage->process(imgUndistorted, imgBinarized);
# if DEBUG
    {
        binarizeStage->show(imgBinarized, "Binarized Image");
    }
#endif
    /*
         Step Three: (detect/estimate lane dividers)

         warp image to birdview

         produce histogram by column on bottom half,
         find two columns (one for left half, the other for right
         half), use both as starting points to mark lane divider
         at bottom of the image.

         run two collumns of sliding windows moving upward,
         each window has fixed width and height.
          collect points falling in each window on the way
          at each stop.

          the centroid of the points in the window serves as new
          center of window at next stop.

          points falling in the two sliding windows
          will be used to determine the second order polynomial.

     */
    auto perspectiveWarpStage = getStage("PerspectiveWarp");
    assert(perspectiveWarpStage);
    cv::Mat warped;
    perspectiveWarpStage->process(img, warped);
    cv::Mat binaryWarped;
    perspectiveWarpStage->process(imgBinarized,binaryWarped);
    auto detectLaneLines = getStage("DetectAndMarkLane");
    assert(detectLaneLines);
   
    cv::Mat frameOut;
    cv::Vec4i leftLine, rightLine;
    detectLaneLines->detectLane(binaryWarped, frameOut, leftLine, rightLine);
    /*
         Step Four: Mark lane

         use the two polynomials to generate vertex points on the two lines.
         fill the lanes and draw the lines, warp them to camera view.

         blend the filled lane and drawn lines in the original images

     */
    cv::Mat blendedFrame;
    detectLaneLines->markLane(img, blendedFrame);

    return false;
}
bool LaneDetectionPipeline::processImages(const vector<cv::Mat>&images)
{
    return false;
}
void LaneDetectionPipeline::sharePipelineReference(shared_ptr<LaneDetectionPipeline> spipeline)
{
    auto iterator = stages.begin();
    while (iterator != stages.end())
    {
        iterator->second->setPipeline(spipeline);
        ++iterator;
    }
}
