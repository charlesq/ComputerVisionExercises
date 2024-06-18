//
//  main.mm
//  LaneDetection
//
//  Created by charles zeng on 6/14/24.
//

#include "CameraCalibration.hpp"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <Foundation/NSString.h>
#include <Foundation/NSDictionary.h>
#include <memory>
#include <assert.h>

using namespace std;

int main(int argc, const char * argv[]) {
    
    shared_ptr<LaneDetectionPipeline> pipeline = make_shared<LaneDetectionPipeline>();
    
    if (!pipeline->createPipeline())
    {
        cerr << "failed to create the pipeline" << endl;
        return -1;
    }
    pipeline->sharePipelineReference(pipeline);
    
    
    cv::Mat img = cv::imread("images/testImg1.png");
    pipeline->processOneFrame(img);

    return 0;
}
