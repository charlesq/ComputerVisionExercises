//
//  LaneDetection.hpp
//  LaneDetection
//
//  Created by charles zeng on 6/15/24.
//  references
//   1. https://github.com/ndrplz/self-driving-car.git
//   2. https://github.com/kipr/opencv/blob/master/modules/contrib/src/polyfit.cpp
//
//
#pragma once
#include <string>
#include <memory>
#include <opencv2/core/utility.hpp>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_map>
#include <stdarg.h>
#include <deque>

using namespace std;

using VARARG = std::variant<cv::Mat*, cv::Vec4i*, int, double>;

class LaneDetectionPipeline;
class CameraCalibrationData
{
    const int resolution[2] = {1920, 1080}; // support only 1920*1080 resolution for the time being
    float fx, fy;
    float ppx, ppy;
    bool isPinhole;
    float distortion[4];
    float rotation[3]; //axis angle vector
    float translation[3];
    cv::Mat warpMatrix;  // perspective warping matrix
    cv::Mat reverseWarpMatrix;
    const cv::Mat& getPerspectiveWarperMatrix() const
    {
        return warpMatrix;
    }
    const cv::Mat& getReversePerspectiveWarperMatrix() const
    {
        return reverseWarpMatrix;
    }
   
    void generatePerspectiveMatrix();
public:
    CameraCalibrationData(const string& plistFile) noexcept(false);
    inline bool isPinholeCamera() const
    {
        return isPinhole;
    }
    void warpPerspective(const cv::Mat& srcFrame, cv::Mat& targetFrame, bool inverse = false) const;
    static shared_ptr<CameraCalibrationData> getCameraCalibrationData() noexcept(false);
    
};

class PipelineStage
{
    const std::string name;
protected:
    std::weak_ptr<PipelineStage> prev;
    std::weak_ptr<PipelineStage> next;
    bool reverseWarp = false;
    cv::Mat _lastFitPixLeft;
    cv::Mat _lastFitPixelRight;
    std::deque<cv::Mat> _recentFitPixelLeft;
    std::deque<cv::Mat> _recentFitPixelRight;
    std::weak_ptr<CameraCalibrationData> camera;
    std::weak_ptr<LaneDetectionPipeline> pipeline;
    
public:
    inline void setReverseWarp(bool v)
    {
        reverseWarp = v;
    }
    inline void setNextStage(std::shared_ptr<PipelineStage> nn)
    {
        next = nn;
    }
    const std::string& getName() const
    {
        return name;
    }
    virtual bool processOneFrame(const cv::Mat& frameRGB)
    {
        return false;
    }
    void setPipeline(std::shared_ptr<LaneDetectionPipeline> spipeline)
    {
        pipeline = spipeline;
    }
    virtual bool markLane(const cv::Mat& frameBinary, cv::Mat& blendedFrame)
    {
        return false;
    }
    PipelineStage(const std::string&& name, std::shared_ptr<PipelineStage> prev):
    name(name),
    prev(prev)
    {
        cv::Mat _lastFitPixLeft = cv::Mat(3, 1, CV_32FC1);
        cv::Mat _lastFitPixelRight = cv::Mat(3, 1, CV_32FC1);
    }
    virtual bool process(const cv::Mat& inFrame, cv::Mat& outFrame)
    {
        return false;
    }
    virtual bool detectLane(cv::Mat &frameBinary, cv::Mat&frameOut, cv::Vec4i& leftLine, cv::Vec4i& rightLine)
    {
        return false;
    }
    PipelineStage(const PipelineStage &&) = delete;
    PipelineStage(const PipelineStage &) = delete;
    PipelineStage& operator = (PipelineStage &) = delete;
    virtual ~PipelineStage() = default;
    void show(const cv::Mat& img, std::string title = "Image") const
    {
        cv::namedWindow(title);
        cv::imshow(title, img);
        cv::waitKey(500);
        cv::destroyWindow(title);
    }
};

class PerspectiveWarp:public virtual PipelineStage
{
    std::weak_ptr<CameraCalibrationData> camera;
public:
    PerspectiveWarp(shared_ptr<CameraCalibrationData>& scamera):
    PipelineStage("PerspectiveWarp", nullptr)
    {
        camera = scamera;
    }
    virtual bool process(const cv::Mat& inFrame, cv::Mat& outFrame)
    {
        if (camera.use_count() == 0) return false;
        camera.lock()->warpPerspective(inFrame, outFrame, reverseWarp);
        return true;
    }
};
class Undistort: public virtual PipelineStage
{
    std::weak_ptr<CameraCalibrationData> camera;
public:
    Undistort(shared_ptr<CameraCalibrationData>& scamera):
    PipelineStage("Undistort", nullptr)
    {
        camera = scamera;
    }
    virtual bool process(const cv::Mat& distorted, cv::Mat& undistored)
    {
        if (camera.use_count() && camera.lock()->isPinholeCamera())
        {
            undistored = distorted;
            return true;
        }
        return false;
    }
};

class Binarize: public virtual PipelineStage
{
    array<float, 3> yellow_HSV_min = {0, 70, 70}; // threshold for HSV binarizing
    array<float, 3> yellow_HSV_max = {50, 255, 255};
protected:
    bool HSVFilter(const cv::Mat& frameRGB, cv::Mat& frameHSVMask);
    bool SobelFilter(const cv::Mat& frameRGB, cv::Mat& frameSobelMask);
    bool HistogramFilter(const cv::Mat& frameRGB, cv::Mat& FrameHist);
    bool MorphFilter(const cv::Mat& frameBinary, cv::Mat& frameMorphed);
    bool EqualizedFilter(const cv::Mat& frame, cv::Mat& frameEqualized);
public:
    Binarize(shared_ptr<PipelineStage> prev):
    PipelineStage("Binarize", prev)
    {
    }
    inline void setYellowHSVThreshold(const array<float, 3>& thresholdMax, const array<float, 3> & thresholdMin)
    {
        yellow_HSV_min = thresholdMin;
        yellow_HSV_max = thresholdMax;
    }
    virtual bool process(const cv::Mat& inFrame, cv::Mat& outFrame);
};

class DetectAndMarkLane: public virtual PipelineStage
{
public:
    void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order);
    DetectAndMarkLane(shared_ptr<PipelineStage> prev):
    PipelineStage("DetectAndMarkLane", prev)
    {
    }
    bool detectLane(cv::Mat &frameBinary, cv::Mat&frameOut, cv::Vec4i& leftLine, cv::Vec4i& rightLine);
    bool markLane(const cv::Mat& frameBinary, cv::Mat& blendedFrame);
    void setMatrixZero(cv::Mat& m) const;
    void drawWarpedLine(cv::Mat& warped, std::vector<cv::Point>& points);
};

class LaneDetectionPipeline
{
public:
    unordered_map<std::string, shared_ptr<PipelineStage>> stages;
    shared_ptr<CameraCalibrationData> camera;
    LaneDetectionPipeline();
    bool createPipeline() noexcept;
    bool processOneFrame(const cv::Mat& frameRGB);
    bool processImages(const vector<cv::Mat>&);
    shared_ptr<PipelineStage> getStage(const std::string& name);
    void sharePipelineReference(shared_ptr<LaneDetectionPipeline> spipline);
    
};

