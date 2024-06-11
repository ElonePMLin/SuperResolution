#ifndef RESOLUTIONMODEL_H
#define RESOLUTIONMODEL_H

#pragma once
#undef slots
#include <torch/script.h>
#define slots Q_SLOTS
#include <opencv2/opencv.hpp>
#define NOMINMAX
#undef min
#undef max

// 模型数据转换函数
// cv::Mat tensor2Mat(at::Tensor& t);

class ResolutionModel {
private:
    torch::jit::script::Module body;
    std::string MODEL_DIR;
    void load();

public:
    ResolutionModel(std::string);
    ~ResolutionModel();
    std::string _MODEL_DIR();
    at::Tensor forward(at::Tensor);
    at::Tensor forward(std::vector<torch::jit::IValue>);
};

#endif // RESOLUTIONMODEL_H
