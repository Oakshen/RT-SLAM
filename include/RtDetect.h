//
// Created by yuwenlu on 2022/3/14.
//
#ifndef RT_DETECT_H
#define RT_DETECT_H

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "RtDetrUltralytics.hpp"
#include "TRTInfer.hpp"
#include <spdlog/spdlog.h>
#include "common.hpp"
using namespace std;

class RtDetection
{
public:
    RtDetection();
    ~RtDetection();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();
    vector<cv::Rect2i> mvPersonArea = {};
    //vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

public:
    cv::Mat mRGB;
    //torch::jit::script::Module mModule;
    std::unique_ptr<RtDetrUltralytics> detector;
    std::unique_ptr<TRTInfer> engine;
    std::vector<std::string> mClassnames;
    // 6-28
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    //cv::Rect2i OpenCV中的矩形类型，表示一个二维的矩形区域 
    map<string, vector<cv::Rect2i>> mmDetectMap;


};


#endif //RT_DETECT_H
