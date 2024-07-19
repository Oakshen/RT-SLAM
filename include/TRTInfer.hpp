#pragma once
#include <NvInfer.h>  // TensorRT API
#include <cuda_runtime_api.h>  // CUDA 运行时 API
#include <fstream>
#include <memory>
#include <vector>
#include <any>
#include <opencv2/opencv.hpp>
#include "Logger.hpp"  // 日志记录器
#include <spdlog/spdlog.h>
using namespace std;

class TRTInfer {
protected:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};  // TensorRT 的 CUDA 引擎共享指针
    nvinfer1::IExecutionContext* context_{nullptr};  // TensorRT 执行上下文指针
    std::vector<void*> buffers_;  // 存储输入和输出缓冲区的指针向量
    nvinfer1::IRuntime* runtime_{nullptr};  // TensorRT 运行时指针
    size_t num_inputs_{0};
    size_t num_outputs_{0};
    static std::shared_ptr<spdlog::logger> logger_;

public:
    TRTInfer(const std::string& model_path);

    void createContextAndAllocateBuffers();
    void initializeBuffers(const std::string& engine_path);
    size_t getSizeByDim(const nvinfer1::Dims& dims);
    void infer();
    std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob);

    ~TRTInfer() {
        for (void* buffer : buffers_) {
            cudaFree(buffer);
        }
    }

    static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) {
        logger_ = logger;
    }
};
