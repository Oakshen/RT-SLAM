#pragma once
#include "common.hpp"
#include "spdlog/spdlog.h"
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect bbox;
    float score;
    int label;
};

class RtDetrUltralytics
{
public:
    RtDetrUltralytics(float confidenceThreshold = 0.25, size_t network_width = 640, size_t network_height = 640);

    cv::Mat preprocess_image(const cv::Mat& image);
    std::vector<Detection> postprocess(const std::vector<std::vector<std::any>>& outputs, const std::vector<std::vector<int64_t>>& shapes, const cv::Size& frame_size);

    static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) 
    {
        logger_ = logger;
    }

private:
    float confidenceThreshold_;
    size_t network_width_;
    size_t network_height_;
    static std::shared_ptr<spdlog::logger> logger_;

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox);
};